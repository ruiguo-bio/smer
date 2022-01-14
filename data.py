import copy

import tables

import math,itertools


import pretty_midi

from vocab import *
import re
import os

TRACK_0_RANGE = (21, 108)


TIME_SIGNATURE_MAX_CHANGE = 1
TEMPO_MAX_CHANGE = 1


MAX_TRACK = 2
V0=120
V1=100
V2=60


def to_category(array, bins):
    result = []
    for item in array:
        result.append(int(np.where((item - bins) >= 0)[0][-1]))
    return result


def select_track_bar_event(event, track_num=None, selected_bars=None):
    bar_pos = np.where(event == 'bar')[0]
    track0_pos = np.where(event == 'track_0')[0]
    track1_pos = np.where(event == 'track_1')[0]
    r = re.compile('i_\d')
    programs = list(filter(r.match, event))
    track_nums = len(programs)

    all_track_event = []

    header_events = event[:2 + track_nums]

    if track_num is None:
        track_num = list(range(track_nums))
    if selected_bars is None:
        selected_bars = list(range(len(bar_pos)))


    if len(track1_pos) > 0:
        bar_track_1 = np.concatenate((bar_pos[1:], [len(event) + 1]))
        track_0_region = np.stack((track0_pos, track1_pos), axis=1)
        track_1_region = np.stack((track1_pos, bar_track_1), axis=1)
    else:
        bar_track_0 = np.concatenate((bar_pos[1:], [len(event) + 1]))
        track_0_region = np.stack((track0_pos, bar_track_0), axis=1)


    track_0_events = []
    track_1_events = []


    for bar in selected_bars:
        region = track_0_region[bar]
        track_0_events.append(['bar'])
        track_0_events[-1].extend(event[region[0]:region[1]])

        if len(track1_pos) > 0:
            region = track_1_region[bar]
            # track_1_events.append(['bar'])
            track_1_events.append(event[region[0]:region[1]])


    track_0_event_copy = copy.deepcopy(track_0_events)
    for idx, events in enumerate(track_0_event_copy):
        all_track_event.append(events)
        if len(track1_pos) > 0:
            all_track_event[-1].extend(track_1_events[idx])

    header_events = header_events.tolist()


    # print(header_events + np.concatenate(all_track_event).tolist())
    pm = event_2midi(header_events + np.concatenate(all_track_event).tolist())
    # print(all_track_event)
    if len(track_num) == 2:
        return all_track_event,pm,pm
    elif 0 in track_num:
        if len(header_events) == 3:
            header_events.pop()
        pm0 = event_2midi(header_events + np.concatenate(track_0_events).tolist())
        return track_0_events, pm,pm0
    else:
        track_1_events = []

        for bar in selected_bars:

            region = track_1_region[bar]
            track_1_events.append(['bar'])
            track_1_events[-1].extend(event[region[0]:region[1]])

        # print(track_1_events)


        track_1_for_pm = np.concatenate(track_1_events).tolist()

        header_events.pop(-2)
        # print(header_events)
        for idx,event in enumerate(track_1_for_pm):
            if event == 'track_1':
                track_1_for_pm[idx] = 'track_0'
        # print(header_events + track_1_for_pm)
        pm1 = event_2midi(header_events + track_1_for_pm)
        return track_1_events, pm,pm1



def remove_drum_track(pm):
    instrument_idx = []
    for idx in range(len(pm.instruments)):
        if pm.instruments[idx].is_drum:
            instrument_idx.append(idx)
    for idx in instrument_idx[::-1]:
        del pm.instruments[idx]
    return pm


def remove_empty_track(pm):
    occupation_rate = []

    beats = pm.get_beats()
    if len(beats) < 20:
        return None

    fs = 4 / (beats[1] - beats[0])

    for instrument in pm.instruments:
        piano_roll = instrument.get_piano_roll(fs=fs)
        if piano_roll.shape[1] == 0:
            occupation_rate.append(0)
        else:
            occupation_rate.append(np.count_nonzero(np.any(piano_roll, 0)) / piano_roll.shape[1])


    for index,rate in enumerate(occupation_rate[::-1]):
        if rate < 0.3:
            pm.instruments.pop(len(occupation_rate) - 1 - index)
    return pm

def get_beat_time(pm, beat_division=4):
    beats = pm.get_beats()

    divided_beats = []
    for i in range(len(beats) - 1):
        for j in range(beat_division):
            divided_beats.append((beats[i + 1] - beats[i]) / beat_division * j + beats[i])
    divided_beats.append(beats[-1])
    divided_beats = np.unique(divided_beats, axis=0)

    beat_indices = []
    for beat in beats:
        beat_indices.append(np.argwhere(divided_beats == beat)[0][0])

    down_beats = pm.get_downbeats()
    down_beats = np.unique(down_beats, axis=0)

    if divided_beats[-1] > down_beats[-1]:
        down_beats = np.append(down_beats, down_beats[-1] - down_beats[-2] + down_beats[-1])

    down_beat_indices = []
    for down_beat in down_beats:
        down_beat_indices.append(np.argmin(np.abs(down_beat - divided_beats)))

    return np.array(divided_beats), np.array(beats), np.array(down_beats), beat_indices, down_beat_indices


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


# files = walk('../dataset/lmd/lmd_separated_melody_bass')
#
# pm = pretty_midi.PrettyMIDI(files[100])



def time2durations(note_duration, duration_time_to_name, duration_times):

    duration_index = np.argmin(np.abs(note_duration - duration_times))
    duration_name = duration_time_to_name[duration_times[duration_index]]
    if duration_name == 'zero':
        return []

    duration_elements = duration_name.split('_')
    return duration_elements


def note_to_event_name(note,duration_time_to_name, duration_times):
    duration_event = time2durations(note.end-note.start, duration_time_to_name, duration_times)

    pitch_event = f'p_{note.pitch}'

    return pitch_event, duration_event


def bar_notes_to_event(notes, bar_time, next_bar_time, beat_times, duration_time_to_name, duration_times,minimum_difference, grid_division=4,is_grid=True):
    bar_event_list = []
    continue_note_dict = {}
    chord_list = []
    in_continue = False
    if len(notes) > 0:
        if is_grid:
            grid_notes(beat_times, notes,minimum_difference,grid_division=grid_division)
        rest_to_bar_start = time2durations(notes[0].start - bar_time, duration_time_to_name, duration_times)

    else:
        rest_to_bar_start = time2durations(next_bar_time - bar_time, duration_time_to_name, duration_times)

    if len(rest_to_bar_start) > 0:
        bar_event_list.append('rest_e')
        bar_event_list.extend(rest_to_bar_start)



    for note in notes:

        if len(chord_list) == 0:
            chord_list.append(note)
        else:
            if math.isclose(note.start,chord_list[-1].start) and math.isclose(note.end,chord_list[-1].end):
                chord_list.append(note)
            else:
                temp_pitch_list = []


                # remove duplicate notes
                chord_list.sort(key=lambda x: x.pitch)
                remove_pos = []
                for pos in range(len(chord_list)-1):
                    if chord_list[pos].pitch == chord_list[pos+1].pitch:
                        remove_pos.append(pos)
                for pos in remove_pos[::-1]:
                    chord_list.pop(pos)

                # clear previous notes in chord_list
                for chord_list_note in chord_list:
                    if chord_list_note.velocity == -1:
                        if not in_continue:
                            temp_pitch_list.append('continue')
                            in_continue = True
                    if chord_list_note.end > next_bar_time:
                        continue_note_for_next_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                                      start=next_bar_time,
                                                                     end=chord_list_note.end,
                                                                      velocity=-1)
                        continue_note_dict[chord_list_note.pitch] = continue_note_for_next_bar

                        note_for_this_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                             start=chord_list_note.start,
                                                             end=next_bar_time,
                                                             velocity=chord_list_note.velocity)

                        pitch_event, duration_event = note_to_event_name(note_for_this_bar,duration_time_to_name, duration_times)

                    else:
                        pitch_event, duration_event = note_to_event_name(chord_list_note,duration_time_to_name, duration_times)

                    temp_pitch_list.append(pitch_event)


                bar_event_list.extend(temp_pitch_list)
                bar_event_list.extend(duration_event)
                in_continue = False

                if note.start >= chord_list[-1].end:
                    # rest relative to previous end
                    rest_duration = time2durations(note.start - chord_list[-1].end, duration_time_to_name,
                                                   duration_times)
                    if len(rest_duration) > 0:
                        bar_event_list.append('rest_e')
                        bar_event_list.extend(rest_duration)

                else:

                    # rest relative to previous start
                    rest_duration = time2durations(note.start - chord_list[-1].start, duration_time_to_name,
                                                   duration_times)
                    bar_event_list.append('rest_s')
                    bar_event_list.extend(rest_duration)
                chord_list = []
                chord_list.append(note)

    else:
        temp_pitch_list = []

        # remove dupliate notes
        chord_list.sort(key=lambda x: x.pitch)
        remove_pos = []
        for pos in range(len(chord_list) - 1):
            if chord_list[pos].pitch == chord_list[pos + 1].pitch:
                remove_pos.append(pos)
        for pos in remove_pos[::-1]:
            chord_list.pop(pos)

        for chord_list_note in chord_list:
            if chord_list_note.velocity == -1:
                if not in_continue:
                    temp_pitch_list.append('continue')
                    in_continue = True
            if chord_list_note.end > next_bar_time:
                continue_note_for_next_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                              start=next_bar_time,
                                                              end=chord_list_note.end,
                                                              velocity=-1)

                continue_note_dict[chord_list_note.pitch] = continue_note_for_next_bar

                note_for_this_bar = pretty_midi.Note(pitch=chord_list_note.pitch,
                                                     start=chord_list_note.start, end=next_bar_time,
                                                     velocity=chord_list_note.velocity)

                pitch_event, duration_event = note_to_event_name(note_for_this_bar,duration_time_to_name, duration_times)

            else:
                pitch_event, duration_event = note_to_event_name(chord_list_note,duration_time_to_name, duration_times)

            temp_pitch_list.append(pitch_event)

        in_continue = False
        if len(temp_pitch_list) > 0:
            bar_event_list.extend(temp_pitch_list)
            bar_event_list.extend(duration_event)

        if chord_list:
            if chord_list_note.end < next_bar_time:
                rest_to_bar_end = time2durations(next_bar_time - chord_list_note.end, duration_time_to_name, duration_times)
                if len(rest_to_bar_end) > 0:
                    bar_event_list.append('rest_e')
                    bar_event_list.extend(rest_to_bar_end)



    return bar_event_list, continue_note_dict


def remove_continue(file_events,is_continue,header_events):



    bar_pos = np.where(file_events == 'bar')[0]
    new_file_events = []

    for idx,event in enumerate(file_events):
        if event == 'continue' and idx<bar_pos[1] and is_continue:
            continue
        else:
            new_file_events.append(event)


    for event in header_events[::-1]:
        new_file_events = np.insert(new_file_events, 0, event)

    if '_' not in new_file_events[1]:
        tempo = float(new_file_events[1])
        tempo_category = int(np.where((tempo - tempo_bins) >= 0)[0][-1])
        new_file_events[1] = f't_{tempo_category}'


    return new_file_events


def grid_notes(beat_times, notes,minimum_difference, grid_division=4):
    divided_beats = []
    for i in range(len(beat_times) - 1):
        for j in range(grid_division):
            divided_beats.append((beat_times[i + 1] - beat_times[i]) / grid_division * j + beat_times[i])
    divided_beats.append(beat_times[-1])

    for note in notes:
        start_grid = np.argmin(np.abs(note.start - divided_beats))

        # maximum note length is two bars
        if note.velocity == -1:
            if note.end > divided_beats[-1]:
                note.end = divided_beats[-1]

        if note.end < divided_beats[-1]+minimum_difference:
            end_grid = np.argmin(np.abs(note.end - divided_beats))
            if start_grid == end_grid:


                if end_grid != len(divided_beats)-1:
                    end_grid += 1
                else:
                    if start_grid != 0:
                        start_grid -= 1
                    else:
                        note.start = -1
                        note.end = -1
                        continue

            note.start = divided_beats[start_grid]
            note.end = divided_beats[end_grid]

        else:
            note.start = divided_beats[start_grid]

    return


def get_note_duration_dict(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
       
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = beat_duration / 3 * 2
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
       
        bar_duration = curr_time_signature[0] * eighth_note_duration

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0

  
    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration


def midi_2event(file_name):

    pm = pretty_midi.PrettyMIDI(file_name)

    pm = remove_drum_track(pm)
    if len(pm.instruments) == 0:
        print('empty track')
        return None
  
    beats = np.unique(pm.get_beats(),axis=0)
    # print(len(beats))

    down_beats = np.unique(pm.get_downbeats(),axis=0)

    if beats[-1] > down_beats[-1]:
        down_beats = np.append(down_beats,down_beats[-1] + down_beats[-1] - down_beats[-2])
    # print(len(down_beats))
    if not math.isclose(down_beats[-1] - beats[-1],0):
        beats = np.append(beats,(beats[-1] + beats[-1] - beats[-2]))
    down_beat_to_beat_indices = []
    for down_beat in down_beats:
        down_beat_to_beat_indices.append(np.argmin(np.abs(beats - down_beat)))


    signatures = [(signature.numerator, signature.denominator) for signature in pm.time_signature_changes]

    ## todo make sure the time signature and tempo change at the start of the bar

    for signature in signatures:
        if signature not in [(4,4),(2,4),(3,4),(6,8)]:
            print(f'not supported signature {signature}, omit {file_name}')
            return None


    signature_change_time = np.array([signature.time for signature in pm.time_signature_changes])

    if signature_change_time[0] != 0:
        print(f'signature change time not at start, omit {file_name}')
        return None

    if len(pm.time_signature_changes) > TIME_SIGNATURE_MAX_CHANGE:
        print(f'more than {TIME_SIGNATURE_MAX_CHANGE} time signature changes, omit {file_name}')
        return None

    tempo_change_times, tempi = pm.get_tempo_changes()

    if tempo_change_times[0] != 0:
        print(f'tempo change time not at start, omit {file_name}')
        return None

    if len(tempo_change_times) > TEMPO_MAX_CHANGE:
        print(f'more than {TEMPO_MAX_CHANGE} tempo changes, omit {file_name}')
        return None

    if signatures[0] == (6,8):
        grid_division = 6
    else:
        grid_division = 4

    event_list = []
    
    track_num = len(pm.instruments)
    track_num = track_num if track_num < MAX_TRACK else MAX_TRACK
    for num in range(track_num):
        pm.instruments[num].notes.sort(key=lambda note: note.start)

    continue_dict_list = []

    for _ in range(track_num):
        continue_dict_list.append({})

    curr_time_signature = signatures[0]
    event_list.append(f'{curr_time_signature[0]}/{curr_time_signature[1]}')

    event_list.append(f'{tempi[0]}')

    for instrument in pm.instruments[:track_num]:
        event_list.append(f'i_{instrument.program}')

    for bar, bar_time in enumerate(down_beats[:-1]):
        event_list.append('bar')

        beat_position = down_beat_to_beat_indices[bar]
        beat_duration = beats[beat_position + 1] - beats[beat_position]

        duration_name_to_time, duration_time_to_name,duration_times, bar_duration = get_note_duration_dict(beat_duration, curr_time_signature)
        minimum_difference = duration_name_to_time['sixteenth'] / 2

        next_bar_time = down_beats[bar + 1]


        for track in range(track_num):
            event_list.append(f'track_{track}')

            # pitch, duration for the next bar if continue
            continue_note_dict = continue_dict_list[track]
            
            note_in_this_bar = [note for note in pm.instruments[track].notes if
                                note.start >= bar_time - minimum_difference and note.start < next_bar_time-minimum_difference]

            for note in note_in_this_bar:
                if note.pitch > TRACK_0_RANGE[1] or note.pitch < TRACK_0_RANGE[0]:
                    print(f"note pitch {note.pitch} out of range, skip this file")
                    return None

            # continue_flag.extend([0] * len(note_in_this_bar))
            beat_in_this_bar = beats[down_beat_to_beat_indices[bar]:down_beat_to_beat_indices[bar+1]+1]
            if len(continue_note_dict.keys()) > 0:
                note_in_this_bar = list(continue_note_dict.values()) + note_in_this_bar

            bar_event_list, continue_note_dict = bar_notes_to_event(note_in_this_bar, bar_time, next_bar_time,beat_in_this_bar,
                                                                    duration_time_to_name, duration_times, minimum_difference,grid_division=grid_division)

            event_list.extend(bar_event_list)
            continue_dict_list[track] = continue_note_dict
    return event_list,pm


def total_duration(duration_list,duration_name_to_time):
    total = 0
    if duration_list:

        for duration in duration_list:
            total += duration_name_to_time[duration]
    return total


def clear_pitch_duration_event(pm_new,
                               track,
                               curr_time,
                               previous_duration,
                               is_rest_s,
                               is_continue,
                               pitch_list,
                               duration_list,
                               duration_name_to_time):
    if is_rest_s:
        duration = total_duration(duration_list,duration_name_to_time)
        curr_time -= previous_duration

    else:
        duration = total_duration(duration_list,duration_name_to_time)

    for pitch in pitch_list:
        if is_continue:
            # look for the previous note, and change the end time of it
            for note in pm_new.instruments[track].notes[::-1]:
                if math.isclose(note.end, curr_time) and note.pitch == pitch:
                    note.end += duration
                    break

        else:
            if track == 0:
                velocity = V0
            elif track == 1:
                velocity = V1
            else:
                velocity = V2
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=curr_time,
                                    end=curr_time + duration)
            pm_new.instruments[track].notes.append(note)

    curr_time += duration
    previous_duration = duration

    return curr_time, previous_duration


def event_2midi(event_list):
    try:

        if event_list[1][0] == 't':
            # print(event_list)
            tempo_category = int(event_list[1][2])
            if tempo_category == len(tempo_bins) -1:
                tempo = tempo_bins[tempo_category]
            else:
                tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category+1]) / 2
        else:
            tempo = float(event_list[1])
        pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        numerator = int(event_list[0].split('/')[0])
        denominator = int(event_list[0].split('/')[1])
        time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
        pm_new.time_signature_changes = [time_signature]



        r = re.compile('i_\d')

        programs = list(filter(r.match, event_list))




        bar_start_pos = np.where('bar' == np.array(event_list))[0][0]


        for track in programs:
            track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
            pm_new.instruments.append(track)


        # add a fake note for duration dict calculation
        pm_new.instruments[0].notes.append(pretty_midi.Note(
            velocity=100, pitch=30, start=0, end=10))
        beats = pm_new.get_beats()
        pm_new.instruments[0].notes.pop()
        duration_name_to_time,duration_time_to_name,duration_times,bar_duration = get_note_duration_dict(beats[1]-beats[0],(time_signature.numerator,time_signature.denominator))

        curr_time = 0

        previous_duration = 0

        in_duration_event = False
        is_rest_s = False
        is_continue = False

        pitch_list = []
        duration_list = []

        bar_num = 0
        track = 0


        sta_dict_list = []
        track_bar_length = []
        track_bar_pitch_length = []
        for _ in programs:
            sta_dict_list.append({'duration_token_length':[],'bar_length':[], 'pitch_token_length':[]})
            track_bar_length.append(0)
            track_bar_pitch_length.append(0)




        current_bar_event = []
        for i, event in enumerate(event_list[bar_start_pos:]):


            current_bar_event.append(event)

            if event in duration_name_to_time.keys():
                duration_list.append(event)
                in_duration_event = True

                track_bar_length[track] += 1

                continue

            if in_duration_event:

                sta_dict_list[track]['duration_token_length'].append(len(duration_list))

                curr_time, previous_duration = clear_pitch_duration_event(pm_new,
                                                                          track,
                                                                          curr_time,
                                                                          previous_duration,
                                                                          is_rest_s,
                                                                          is_continue,
                                                                          pitch_list,
                                                                          duration_list,
                                                                          duration_name_to_time)


                pitch_list = []
                duration_list = []

                in_duration_event = False
                is_rest_s = False
                is_continue = False


            pitch_match = re.search(r'p_(\d+)', event)
            if pitch_match:

                track_bar_pitch_length[track] += 1

                pitch = int(pitch_match.group(1))
                pitch_list.append(pitch)

            if event == 'rest_s':
                is_rest_s = True

            if event == 'continue':
                is_continue = True


            if event == 'bar':
                bar_start_time = bar_num * bar_duration
                bar_num += 1

                if bar_num != 1:

                    for i in range(len(programs)):
                        sta_dict_list[i]['bar_length'].append(track_bar_length[i])
                        sta_dict_list[i]['pitch_token_length'].append(track_bar_pitch_length[i])
                        track_bar_length[i] = track_bar_pitch_length[i] = 0

                current_bar_event = []
                continue



            track_match = re.search(r'track_(\d)', event)

            if track_match:
                curr_time = bar_start_time
                previous_duration = 0
                track = int(track_match.group(1))

            track_bar_length[track] += 1

        return pm_new
    except Exception as e:
        print(f'exceptions {e}')
        return None


#



def add_duration(duration_list,current_step):
    total_duration = 0
    for duration in duration_list:
        total_duration += duration_to_time[duration]
    total_duration = int(total_duration*4)
    last_time = int(current_step[2:])
    current_step = f'e_{last_time + total_duration}'
    if total_duration > 32:
        print(f'total duration {total_duration}')

    total_duration = f'n_{total_duration}'
    return total_duration, current_step


def remi_2midi(events):
    if events[1][0] == 't':
        # print(events)
        tempo_category = int(events[1][2])
        if tempo_category == len(tempo_bins) - 1:
            tempo = tempo_bins[tempo_category]
        else:
            tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category + 1]) / 2
    else:
        tempo = float(events[1])
    pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    numerator = int(events[0].split('/')[0])
    denominator = int(events[0].split('/')[1])
    time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
    pm_new.time_signature_changes = [time_signature]

    r = re.compile('i_\d')

    programs = list(filter(r.match, events))

  
    for track in programs:
        track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
        pm_new.instruments.append(track)

    # add a fake note for duration dict calculation
    pm_new.instruments[0].notes.append(pretty_midi.Note(
        velocity=100, pitch=30, start=0, end=10))
    beats = pm_new.get_beats()
    pm_new.instruments[0].notes.pop()
    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
        beats[1] - beats[0], (time_signature.numerator, time_signature.denominator))
    sixteenth_duration = duration_name_to_time['sixteenth']
    curr_time = 0
    bar_num = 0
    bar_start_time = 0
    pitch_list = []
    for idx, event in enumerate(events):
        # print(idx,event)
        if event == 'bar':
            curr_time = bar_num * bar_duration
            bar_start_time = curr_time
            bar_num += 1
        if event in track_num:
            curr_time = bar_start_time
            current_track = event

        if event in step_token:
            curr_time = bar_start_time + int(event[2:]) * sixteenth_duration
        if event in pitch_tokens:
            pitch_list.append(int(event[2:]))

        if event in duration_single:
            end_time = curr_time + (int(event[2:])) * sixteenth_duration
            start_time = curr_time
            for pitch in pitch_list:
                if current_track == 'track_0':
                    vel = V0
                elif current_track == 'track_1':
                    vel = V1
                else:
                    vel = V2
                note = pretty_midi.Note(velocity=vel, pitch=pitch,
                                        start=start_time, end=end_time)
                pm_new.instruments[int(current_track[6])].notes.append(note)
            pitch_list = []

    return pm_new

def rest_multi_step_single(events, remove_rest=True,remove_continue=True):

    in_duration = False
    in_pitch = False
    is_rest_s = False

    new_event = []
    is_continue = False
    current_step = 'e_0'
    pitch_list = []
    curr_track = ''
    bar_num = 0
    r = re.compile('track_\d')

    track_program = list(set(filter(r.match, events)))
    track_nums = len(track_program)


    duration_list = []

    previous_step = 'e_0'
    for idx,event in enumerate(events):

        if event not in duration_multi and in_duration:

            if is_rest_s and new_event[-1] in pitch_tokens:
                total_duration, _ = add_duration(duration_list, current_step)
            elif is_rest_s and is_continue:
                total_duration, current_step = add_duration(duration_list, previous_step)
            elif is_rest_s and new_event[-1] in duration_single + track_num:
                current_step = previous_step
                total_duration, current_step = add_duration(duration_list, current_step)
            else:
                previous_step = current_step
                total_duration, current_step = add_duration(duration_list, current_step)

            is_rest_s = False

            in_duration = False
            duration_list = []
            if in_pitch:
                if int(total_duration[2:]) > 32:
                    print(f'total duration is {total_duration}')
                new_event.append(total_duration)
                in_pitch = False
            if is_continue:
                track_pos = np.where(np.array(new_event) == curr_track)[0][-2]
                bar_pos = np.where(np.array(new_event) == 'bar')[0][-1]
                if curr_track == 'track_0':
                    if track_nums > 1:
                        next_track_pos = np.where(np.array(new_event) == 'track_1')[0][-1]
                    else:
                        next_track_pos = bar_pos
                elif curr_track == 'track_1':
                    if track_nums > 2:
                        next_track_pos = np.where(np.array(new_event) == 'track_2')[0][-1]
                    else:
                        next_track_pos = bar_pos
                else:
                    next_track_pos = bar_pos

                for pitch in pitch_list:
                    if len(np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0]) > 0:

                        pitch_pos = track_pos + np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0][-1]

                        for token in new_event[pitch_pos + 1:]:
                            if token in duration_single:
                                break
                        old_duration = token


                        for token in new_event[pitch_pos - 1:track_pos:-1]:
                            if token in step_token:
                                break
                        old_step = token


                        if new_event[pitch_pos-1] in step_token:
                            if new_event[pitch_pos+1] in duration_single:
                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event[pitch_pos+1] = new_duration
                            else:

                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                new_event.insert(pitch_pos + 1, old_step)
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event.insert(pitch_pos+1,new_duration)
                                next_track_pos += 2

                        else:
                            new_event.insert(pitch_pos, old_step)
                            new_event.insert(pitch_pos, old_duration)
                            next_track_pos += 2
                            if new_event[pitch_pos + 3] in duration_single:
                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event[pitch_pos + 3] = new_duration
                            else:

                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event.insert(pitch_pos + 3, old_step)
                                new_event.insert(pitch_pos + 3, new_duration)
                                next_track_pos += 2

                        pop_list = []
                        total_break = False
                        for pos in range(track_pos,next_track_pos):
                            if total_break:
                                break
                            if new_event[pos] in step_token:
                                for duration_pos in range(pos+1,next_track_pos):
                                    if new_event[duration_pos] in duration_single:
                                        this_duration = new_event[duration_pos]
                                        break

                                for next_pos in range(pos+1,next_track_pos):
                                    if total_break:
                                        break
                                    if new_event[next_pos] in step_token:
                                        if new_event[next_pos] == new_event[pos]:
                                            for next_duration_pos in range(next_pos + 1, next_track_pos):
                                                if new_event[next_duration_pos] in duration_single:
                                                    next_duration = new_event[next_duration_pos]
                                                    if next_duration == this_duration:
                                                        if next_pos-1 != duration_pos:
                                                            continue

                                                            for move_pitch_pos in range(next_duration_pos-1,next_pos,-1):
                                                                new_event.insert(duration_pos, new_event[move_pitch_pos])
                                                                del new_event[move_pitch_pos+1]
                                                            pop_list.append(next_pos+1)
                                                            pop_list.append(next_duration_pos)
                                                            total_break = True
                                                            break


                                                        else:
                                                            pop_list.append(duration_pos)
                                                            pop_list.append(next_pos)
                                                    break
                        if len(pop_list):
                            for pop_pos in pop_list[::-1]:
                                del new_event[pop_pos]
                            next_track_pos -= len(pop_list)
                            
                is_continue = False
                pitch_list = []

        if event in rests:
            if event == 'rest_s':
               
                is_rest_s = True

            if remove_rest:
                pass
            else:
                new_event.append(event)
            continue

        if event in track_num:
            current_step = 'e_0'
            previous_step = 'e_0'
            duration_list = []
            pitch_list = []
            in_duration = False
            in_pitch = False
            is_rest_s = False
            is_continue = False
            new_event.append(event)
            curr_track = event
            continue

        if event in pitch_tokens:
            if is_continue:
                pitch_list.append(event)
            else:
                if not in_pitch:
                    if is_rest_s:
                        if int(previous_step[2:]) > 15:
                            print(f'previous step is {previous_step}')
                        new_event.append(previous_step)
                        current_step = previous_step
                        is_rest_s = False
                    else:
                        if int(current_step[2:]) > 15:
                            print(f'current step is {current_step}')
                        new_event.append(current_step)
                    in_pitch = True
                new_event.append(event)

            continue

        if event in duration_multi:
            duration_list.append(event)
            in_duration = True
            continue

        if event == 'continue':

            is_continue = True
            if remove_continue:
                pass
            else:
                new_event.append(event)
            continue

        if event == 'bar':
            bar_num += 1

            # print(bar_num)
            # if bar_num == 90:
            #     print("here")

        new_event.append(event)


    else:
        if is_rest_s and new_event[-1] in pitch_tokens:
            total_duration, _ = add_duration(duration_list, current_step)
        elif is_rest_s and is_continue:
            total_duration, current_step = add_duration(duration_list, previous_step)
        elif is_rest_s and new_event[-1] in duration_single + track_num:
            current_step = previous_step
            total_duration, current_step = add_duration(duration_list, current_step)
        else:
            previous_step = current_step
            total_duration, current_step = add_duration(duration_list, current_step)
        is_rest_s = False

        in_duration = False
        duration_list = []
        if in_pitch:
            new_event.append(total_duration)
            in_pitch = False
        if is_continue:
            track_pos = np.where(np.array(new_event) == curr_track)[0][-2]
            bar_pos = np.where(np.array(new_event) == 'bar')[0][-1]
            if curr_track == 'track_0':
                if track_nums > 1:
                    next_track_pos = np.where(np.array(new_event) == 'track_1')[0][-1]
                else:
                    next_track_pos = bar_pos
            elif curr_track == 'track_1':
                if track_nums > 2:
                    next_track_pos = np.where(np.array(new_event) == 'track_2')[0][-1]
                else:
                    next_track_pos = bar_pos
            else:
                next_track_pos = bar_pos

            for pitch in pitch_list:
                if len(np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0]) > 0:
                    pitch_pos = track_pos + np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0][-1]

                    for token in new_event[pitch_pos + 1:]:
                        if token in duration_single:
                            break
                    old_duration = token


                    for token in new_event[pitch_pos - 1:track_pos:-1]:
                        if token in step_token:
                            break
                    old_step = token

                    if new_event[pitch_pos - 1] in step_token:
                        if new_event[pitch_pos + 1] in duration_single:
                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event[pitch_pos + 1] = new_duration
                        else:

                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event.insert(pitch_pos + 1, old_step)
                            new_event.insert(pitch_pos + 1, new_duration)
                            next_track_pos += 2

                    else:
                        new_event.insert(pitch_pos, old_step)
                        new_event.insert(pitch_pos, old_duration)
                        next_track_pos += 2
                        if new_event[pitch_pos + 3] in duration_single:
                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event[pitch_pos + 3] = new_duration
                        else:

                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event.insert(pitch_pos + 3, old_step)
                            new_event.insert(pitch_pos + 3, new_duration)
                            next_track_pos += 2

                    pop_list = []
                    total_break = False
                    for pos in range(track_pos, next_track_pos):
                        if total_break:
                            break
                        if new_event[pos] in step_token:
                            for duration_pos in range(pos + 1, next_track_pos):
                                if new_event[duration_pos] in duration_single:
                                    this_duration = new_event[duration_pos]
                                    break

                            for next_pos in range(pos + 1, next_track_pos):
                                if total_break:
                                    break
                                if new_event[next_pos] in step_token:
                                    if new_event[next_pos] == new_event[pos]:
                                        for next_duration_pos in range(next_pos + 1, next_track_pos):
                                            if new_event[next_duration_pos] in duration_single:
                                                next_duration = new_event[next_duration_pos]
                                                if next_duration == this_duration:
                                                    if next_pos - 1 != duration_pos:
                                                        continue

                                                        for move_pitch_pos in range(next_duration_pos - 1, next_pos, -1):
                                                            new_event.insert(duration_pos, new_event[move_pitch_pos])
                                                            del new_event[move_pitch_pos + 1]

                                                        pop_list.append(next_pos + 1)
                                                        pop_list.append(next_duration_pos)
                                                        total_break = True
                                                        break


                                                    else:
                                                        pop_list.append(duration_pos)
                                                        pop_list.append(next_pos)
                                                break
                    if len(pop_list):
                        for pop_pos in pop_list[::-1]:
                            del new_event[pop_pos]
                        next_track_pos -= len(pop_list)


    # print(events)
    # print(new_event)
    return new_event


def cal_features_diffs(pm_generated,pm_original):
    features_generated = cal_features(pm_generated)
    feature_original = cal_features(pm_original)
    features_diffs = {}

    # print(features_generated)
    #
    # print(feature_original)

    for key in features_generated.keys():

        #     print(f'the number of items is {len(rest_multi_generated[key])}')
        if 'hist' not in key and 'chro' not in key:
            if feature_original[key] == 0:
                print(f'original {key} is 0, omit')
                features_diffs[key] = 0
            else:
                features_diffs[key] = np.abs(
                    features_generated[key] - feature_original[key]) / feature_original[key]
        else:
            if np.sum(
                np.square(feature_original[key])) == 0:
                    print(f'original {key} is 0, omit')
                    features_diffs[key] = 0
            else:
                features_diffs[key] = np.sum(
                    np.square(features_generated[key] - feature_original[key])) / np.sum(
                    np.square(feature_original[key]))

    return features_diffs

def cal_features(pm):
    """
    only one track in pm, all bars are calculated

    Returns:
    used_pitch
    used_note
    pitch_histogram
    pitch_interval_hist # not for track 2
    pitch_range
    onset_interval_hist
    duration_hist
    """
    result_features = {}
    chromagram = np.zeros(12)
    duration_hist = np.zeros(32)
    pitch_intervals_hist = np.zeros(12)
    onset_interval_hist = np.zeros(32)

    track_total_notes = pm.instruments[0].notes
    used_pitch = set()
    total_notes = []

    duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_note_duration_dict(
        pm.get_beats()[1]-pm.get_beats()[0], (pm.time_signature_changes[0].numerator, pm.time_signature_changes[0].denominator))


    sixteenth_time = round(duration_name_to_time['sixteenth'],2)
    bar_time = list(pm.get_downbeats())
    bar_time.append(bar_time[-1] + bar_duration)

    for note in track_total_notes:

        used_pitch.add(note.pitch)
        chromagram[note.pitch % 12] += 1
        if len(total_notes) > 0:
            if 0 <= round(note.start - total_notes[-1].end,2) <= 2:
                intervals = abs(note.pitch - total_notes[-1].pitch)
                if intervals > 11:
                    continue
                pitch_intervals_hist[intervals] += 1
            if sixteenth_time <= round(note.start - total_notes[-1].start,2) <= sixteenth_time * 32 :
                onset_interval_hist[int(round(note.start - total_notes[-1].start,2) / sixteenth_time)-1] += 1

        total_notes.append(note)

        duration = int(round(note.end - note.start,2) / sixteenth_time)
        # print(round(note.end - note.start, 2))
        # print(duration)
        if duration > 32:
            duration = 32
        duration_hist[duration-1] += 1

    used_pitch_number = len(used_pitch)
    if used_pitch_number == 0:
        return None
    chromagram = chromagram / sum(chromagram)
    pitch_range = max(used_pitch) - min(used_pitch)
    used_notes_number = len(total_notes)
    pitch_intervals_hist = pitch_intervals_hist / sum(pitch_intervals_hist)
    duration_hist = duration_hist / sum(duration_hist)
    onset_interval_hist = onset_interval_hist / sum(onset_interval_hist)

    result_features['pitch_number'] = used_pitch_number
    result_features['note_number'] = used_notes_number
    result_features['pitch_range'] = pitch_range
    result_features['chromagram'] = chromagram
    result_features['pitch_intervals_hist'] = pitch_intervals_hist
    result_features['duration_hist'] = duration_hist
    result_features['onset_interval_hist'] = onset_interval_hist


    return result_features




#
def walk(folder_name,suffix):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-len(suffix):] == suffix:
                files.append(os.path.join(p, file_name))
    return files



def create_input(file_events):
    return_list = []

    r = re.compile('i_\d')
    track_program = list(filter(r.match, file_events))
    num_of_tracks = len(track_program)

    header_events = file_events[:2 + num_of_tracks]

    bar_pos = np.where(file_events == 'bar')[0]
    bar_beginning_pos = bar_pos[::16]

    for pos in range(len(bar_beginning_pos)):
        if pos == 0:
            is_continue = True
        else:
            is_continue = False

        if pos == len(bar_beginning_pos) - 1:
            return_events = remove_continue(file_events[bar_beginning_pos[pos]:], is_continue, header_events)
        else:
            return_events = remove_continue(file_events[bar_beginning_pos[pos]:bar_beginning_pos[pos + 1]],
                                                 is_continue, header_events)

        if return_events is not None:
            return_list.append(return_events.tolist())

    print(f'number of data of this song is {len(return_list)}')
    return return_list

# #
# vocab = WordVocab(all_tokens)
# event_folder = '/Users/ruiguo/Downloads/score_transformer/jay_event'

# event_folder = '/home/ruiguo/dataset/lmd/lmd_event_corrected_0723/'
# event_folder = '/home/data/guorui/dataset/lmd/only_melody_bass_event'
#
# # # event_folder = './dataset/lmd_event_corrected_0723/'
# # # event_folder = '/home/ruiguo/dataset/chinese_event'
# files_sheet = walk(event_folder,suffix='event')
# files_remi = walk(event_folder,suffix='step_single')
# files_step_multi = walk(event_folder,suffix='step_multi')
# files_rest_single = walk(event_folder,suffix='rest_single')
# # files_chinese = walk('/home/ruiguo/dataset/chinese/event',suffix='event')
# # print(len(files_chinese))
# # assert len(files_sheet) == len(files_remi) == len(files_step_multi) == len(files_rest_single)
# # # # #



#
#
# # #
# rest_multi = False
# #
# create_test = True
# logger = logging.getLogger(__name__)
#
# logger.handlers = []
#
# add_control = False
#
# if create_test:
#     if add_control:
#         if rest_multi:
#             logfile = 'rest_multi_control_test.log'
#         else:
#             logfile = 'step_single_control_test.log'
#     else:
#         if rest_multi:
#             logfile = 'rest_multi_test.log'
#         else:
#             logfile = 'step_single_test.log'
# else:
#     if add_control:
#         if rest_multi:
#             logfile = 'dataset_rest_multi_all_control_training_augment.log'
#         else:
#             logfile = 'dataset_step_single_all_control_training_augment.log'
#     else:
#         if rest_multi:
#             logfile = 'dataset_rest_multi_training_augment.log'
#         else:
#             logfile = 'dataset_step_single_training_augment.log'
#
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
#                         datefmt='%Y-%m-%d %H:%M:%S', filename=logfile, filemode='w')
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s',
#                               datefmt='%Y-%m-%d %H:%M:%S')
# console.setFormatter(formatter)
# logger.addHandler(console)
#
# coloredlogs.install(level='INFO', logger=logger, isatty=True)


# #
# for idx,file_name in enumerate(files):
#     cal_separate_file(files,idx)
# keydata = json.load(open(event_folder + '/keys.json','r'))
#

#
# load a model for key prediction

#
# checkpoint_epoch = 21
# config_folder = '/home/data/guorui/wandb/run-20210423_094640-sw0lyk9u/'
# folder_prefix = '/home/ruiguo/'
# with open(os.path.join(config_folder,"files/config.yaml")) as file:
#
#     config = yaml.full_load(file)
#
#
# vocab = WordVocab(all_tokens)
# model_prediction = ScoreTransformer(vocab.vocab_size, config['d_model']['value'], config['nhead']['value'], config['num_encoder_layers']['value'],
#                                  config['num_encoder_layers']['value'], 2048, 2400,
#                                  0.1, 0.1)
#
# model_prediction_dict = torch.load(os.path.join(config_folder,"files/checkpoint_21"))
# model_prediction_state = model_prediction_dict['model_state_dict']
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# from collections import OrderedDict
# new_prediction_state_dict = OrderedDict()
# for k, v in model_prediction_state.items():
#     name = k[7:]  # remove `module.`
#     new_prediction_state_dict[name] = v
#
# # new_state_dict = model_state
#
# model_prediction.load_state_dict(new_prediction_state_dict)
# model_prediction.to(device)
#
# create test data
# if rest_multi:
#     files = files_sheet
# else:
#     files = files_remi
# #
# #
# if create_test:
#     total_events = []
#     total_names = []
#     start_num = int(len(files) * .9)
#     end_num = int(len(files) * 1)
#     print(f'start number {start_num} end number {end_num}')
#     for idx,file_name in enumerate(files[start_num:]):
#         events = cal_separate_file(files,idx,augment=False,add_control=add_control,rest_multi=rest_multi)
#         if events:
#             total_events.append(events)
#             h5_file_name = '/home/ruiguo/dataset/lmd/lmd_matched_h5/' + '/'.join(file_name.split('/')[7:-1]) + '.h5'
#             with tables.open_file(h5_file_name) as h5:
#                 print((h5.root.metadata.songs.cols.title[0],
#                                     h5.root.metadata.songs.cols.artist_name[0]))
#                 total_names.append((h5.root.metadata.songs.cols.title[0],
#                                     h5.root.metadata.songs.cols.artist_name[0]))
#
#
#     if rest_multi:
#         pickle.dump(total_events, open(f'/home/data/guorui/score_transformer/sync/rest_multi_no_control_test_batches', 'wb'))
#         pickle.dump(total_names,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_no_control_test_batch_names', 'wb'))
#     else:
#         pickle.dump(total_events, open(f'/home/data/guorui/score_transformer/sync/step_single_no_control_test_batches', 'wb'))
#         pickle.dump(total_names,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_no_control_test_batch_names', 'wb'))
#

import shutil

# sheet_more = 0
# remi_more = 0
#
# sheet_more_number = 0
# remi_more_number = 0
# sheet_0 = pickle.load(open(files_sheet[0], 'rb'))
# remi_0 = pickle.load(open(files_remi[0], 'rb'))
# for i in range(25881):
#     sheet = pickle.load(open(files_sheet[i], 'rb'))
#     remi = pickle.load(open(files_remi[i], 'rb'))
#     if len(sheet) > len(remi):
#         sheet_more_number += len(sheet) - len(remi)
#     else:
#         remi_more_number += len(remi) - len(sheet)
#
#
#     sheet_total += os.path.getsize(one_file)
#     shutil.copy(one_file,'/home/ruiguo/dataset/all_sheet')

#
#
#

# 
# 
# start_num = int(len(files) * .9)
# end_num = int(len(files) * .91)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')

# #
# # mock_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batches', 'rb'))
# # mock_3 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batches', 'rb'))
# #
# # mock_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batch_lengths', 'rb'))
# #
# # mock_2 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batch_lengths', 'rb'))
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_mock_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_mock_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_mock_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_mock_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_mock_batch_lengths',
#                          'wb'))
#
#
# pickle.dump(training_all_batches, open(f'./dataset/rest_multi_augment_all_control_mock_batches', 'wb'))
# pickle.dump(training_batch_length,
#             open(f'./dataset/rest_multi_augment_all_control_mock_batch_lengths', 'wb'))
# #
# #
# start_num = int(len(files) * .8)
# end_num = int(len(files) * .9)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
# #
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batch_lengths',
#                          'wb'))
#
# start_num = int(len(files) * .9)
# end_num = int(len(files) * 1)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
# 
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
# 
# 
# #
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_test_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_test_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_test_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_test_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_test_batch_lengths',
#                          'wb'))
# #
# # #
#
# start_num = int(len(files) * 0)
# end_num = int(len(files) * .8)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batch_lengths_0', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_0',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batch_lengths_0',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batches_0', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batch_lengths_0',
#                          'wb'))



# #
# start_num = int(len(files) * .6)
# end_num = int(len(files) * .8)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=True,add_control=add_control,rest_multi=rest_multi)
#
#

# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_training_batch_lengths_2', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_2',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_training_batch_lengths_2',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batches_2', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_training_batch_lengths_2',
#                          'wb'))
#




#
# start_num = int(len(files) * .8)
# end_num = int(len(files) * .9)
# # end_num = len(files)
# print(f'start file num is {start_num}')
# print(f'end file num is {end_num}')
#
# training_all_batches, training_batch_length = gen_batches(files[start_num:end_num],augment=False,add_control=add_control,rest_multi=rest_multi)
#
# if add_control:
#     if rest_multi:
#         pickle.dump(training_all_batches, open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                 open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_all_control_validation_batch_lengths', 'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_validation_batch_lengths',
#                          'wb'))
# else:
#     if rest_multi:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/rest_multi_augment_validation_batch_lengths',
#                          'wb'))
#     else:
#         pickle.dump(training_all_batches,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batches', 'wb'))
#         pickle.dump(training_batch_length,
#                     open(f'/home/data/guorui/score_transformer/sync/step_single_augment_validation_batch_lengths',
#                          'wb'))
#


#
#
# if add_control:
#     training_all_batches_0 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batches_0', 'rb'))
#     training_batch_length_0 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths_0', 'rb'))
#
#
#     training_all_batches_1 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batches_1', 'rb'))
#     training_batch_length_1 = pickle.load(open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths_1', 'rb'))
# else:
#     training_all_batches_0 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_0', 'rb'))
#     training_batch_length_0 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_0', 'rb'))
#
#     training_all_batches_1 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_1', 'rb'))
#     training_batch_length_1 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_1', 'rb'))
#
#     training_all_batches_2 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batches_2', 'rb'))
#     training_batch_length_2 = pickle.load(
#         open(f'./dataset/rest_multi_augment_training_batch_lengths_2', 'rb'))
#     #
#
# length_0 = len(training_all_batches_0)
# length_1 = len(training_all_batches_1)
# training_all_batches_0.extend(training_all_batches_1)
# training_all_batches_0.extend(training_all_batches_2)
#
# length_1_shifted = copy.copy(training_batch_length_1)
# length_2_shifted = copy.copy(training_batch_length_2)
#
# for key1,values1 in length_1_shifted.items():
#     values = [value + length_0 for value in values1]
#     length_1_shifted[key1] = values
#
#
# for key1, values1 in length_1_shifted.items():
#     if key1 in training_batch_length_0:
#         training_batch_length_0[key1].extend(values1)
#     else:
#         training_batch_length_0[key1] = values1
#
#
#
#
# for key2,values2 in length_2_shifted.items():
#     values = [value + length_0 + length_1 for value in values2]
#     length_2_shifted[key2] = values
#
#
# for key2, values2 in length_2_shifted.items():
#     if key2 in training_batch_length_0:
#         training_batch_length_0[key2].extend(values2)
#     else:
#         training_batch_length_0[key2] = values2
#
#
#
# total_length = 0
# for key,values in training_batch_length_0.items():
#     total_length += len(values)
#
#
# if add_control:
#     pickle.dump(training_all_batches_0, open(f'./dataset/rest_multi_augment_all_control_training_batches', 'wb'))
#     pickle.dump(training_batch_length_0,
#                 open(f'./dataset/rest_multi_augment_all_control_training_batch_lengths', 'wb'))
#
# else:
#     pickle.dump(training_all_batches_0,
#                 open(f'./dataset/rest_multi_augment_two_track_training_batches', 'wb'))
#     pickle.dump(training_batch_length_0,
#                 open(f'./dataset/rest_multi_augment_two_track_training_batch_lengths',
#                      'wb'))

#
#
# training_all_batches_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_0', 'rb'))
# training_batch_length_0 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_0', 'rb'))
#
#
# training_all_batches_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches_1', 'rb'))
# training_batch_length_1 = pickle.load(open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths_1', 'rb'))
# #
# training_all_batches_0.extend(training_all_batches_1)
#
# length_1_shifted = copy.copy(training_batch_length_1)
#
# for key,values in length_1_shifted.items():
#     values = [value + len(training_all_batches_0) - len(training_all_batches_1) for value in values]
#     length_1_shifted[key] = values
#
#
# for key1, values1 in length_1_shifted.items():
#     if key1 in training_batch_length_0:
#         training_batch_length_0[key1].extend(values1)
#     else:
#         training_batch_length_0[key1] = values1
#
#
# total_length = 0
# for key,values in training_batch_length_0.items():
#     total_length += len(values)
#
#
#
# pickle.dump(training_all_batches_0, open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batches', 'wb'))
# pickle.dump(training_batch_length_0,
#             open(f'/home/data/guorui/score_transformer/sync/step_single_augment_all_control_training_batch_lengths', 'wb'))



# print('')


#
#


