import numpy as np
import data
from einops import rearrange
import torch
import re
import vocab

def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


def sampling(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_rest=False,no_whole_duration=False,no_eos=False,no_continue=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_continue:
        logit = np.array([-100 if i == vocab.continue_index else logit[i] for i in range(vocab.vocab_size)])

    if no_rest:
        logit = np.array([-100 if i in vocab.rest_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_whole_duration:
        logit = np.array([-100 if i == vocab.duration_only_indices[0] else logit[i] for i in range(vocab.vocab_size)])

    if no_eos:
        logit = np.array([-100 if i == vocab.eos_index else logit[i] for i in range(vocab.vocab_size)])

    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word



def sampling_rest_single(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_rest=False,no_eos=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])


    if no_rest:
        logit = np.array([-100 if i in vocab.rest_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_eos:
        logit = np.array([-100 if i == vocab.eos_index else logit[i] for i in range(vocab.vocab_size)])

    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


def sampling_step_single(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_step=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_step:
        logit = np.array([-100 if i in vocab.step_indices else logit[i] for i in range(vocab.vocab_size)])


    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word



def sampling_step_multi(logit,vocab, p=None, t=1.0,no_pitch=False,no_duration=False,no_step=False,no_eos=False,no_continue=False):
    logit = logit.squeeze().cpu().numpy()
    if no_pitch:

        logit = np.array([-100 if i in vocab.pitch_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_duration:
        logit = np.array([-100 if i in vocab.duration_only_indices else logit[i] for i in range(vocab.vocab_size)])

    if no_step:
        logit = np.array([-100 if i in vocab.step_indices else logit[i] for i in range(vocab.vocab_size)])
    if no_eos:
        logit = np.array([-100 if i == vocab.eos_index else logit[i] for i in range(vocab.vocab_size)])
    if no_continue:
        logit = np.array([-100 if i == vocab.continue_index else logit[i] for i in range(vocab.vocab_size)])

    logit = np.array([-100 if i in vocab.program_indices + vocab.structure_indices + vocab.time_signature_indices + vocab.tempo_indices  else logit[i] for i in range(vocab.vocab_size)])

    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word



def gen_nopeek_mask(length):
    """
     Returns the nopeek mask
             Parameters:
                     length (int): Number of tokens in each sentence in the target batch
             Returns:
                     mask (arr): tgt_mask, looks like [[0., -inf, -inf],
                                                      [0., 0., -inf],
                                                      [0., 0., 0.]]
     """
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def model_generate(model, src, tgt,device,return_weights=False):

    src = src.clone().detach().unsqueeze(0).long().to(device)
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    tgt_mask = gen_nopeek_mask(tgt.shape[1])
    tgt_mask = tgt_mask.clone().detach().unsqueeze(0).to(device)


    output,weights = model.forward(src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                           tgt_mask=tgt_mask)
    if return_weights:
        return output.squeeze(0).to('cpu'), weights.squeeze(0).to('cpu')
    else:
        return output.squeeze(0).to('cpu')

def mask_bar_and_track(event, vocab, tracks_to_generate='all', bars_to_generate='all'):

    tokens = []
    r = re.compile('i_\d')

    track_program = list(filter(r.match, event))
    track_nums = len(track_program)
    bar_poses = np.where(np.array(event) == 'bar')[0]

    if tracks_to_generate == 'all':
        mask_tracks = list(range(track_nums))
    else:
        mask_tracks = [int(item) for item in tracks_to_generate.split(',')]


    if bars_to_generate == 'all':
        mask_bars = list(range(len(bar_poses)))
    else:
        mask_bars = [int(item) for item in bars_to_generate.split(',')]

    decoder_target = []
    masked_indices_pairs = []
    mask_bar_names = []
    mask_track_names = []


    track_end_poses = []

    track_0_pos = np.where('track_0' == np.array(event))[0]
    track_1_pos = np.where('track_1' == np.array(event))[0]
    for pos in track_1_pos[:-1]:
        track_end_poses.append(bar_poses[np.where(pos < bar_poses)[0][0]])
    else:
        track_end_poses.append(len(event))
    all_track_pos = np.sort(np.concatenate([track_0_pos, track_1_pos, track_end_poses]))
    bar_with_track_poses = []

    for i, pos in enumerate(all_track_pos):
        if i % (track_nums + 1) == 0:
            this_bar_poses = []
            this_bar_pairs = []
            this_bar_poses.append(pos)

        else:
            this_bar_poses.append(pos)
            if i % (track_nums + 1) == track_nums:
                for j in range(len(this_bar_poses) - 1):
                    this_bar_pairs.append((this_bar_poses[j], this_bar_poses[j + 1]))

                bar_with_track_poses.append(this_bar_pairs)



    for mask_bar in mask_bars:

        track_mask_poses = mask_tracks

        for track_mask_pos in track_mask_poses:
            mask_track_names.append(track_mask_pos)
            mask_bar_names.append(mask_bar)
            bar_with_track_poses[mask_bar][track_mask_pos]
            masked_indices_pairs.append(bar_with_track_poses[mask_bar][track_mask_pos])

    token_events = event.copy()

    for masked_pairs in masked_indices_pairs:
        masked_token = event[masked_pairs[0]:masked_pairs[1]]

        for token in masked_token:
            decoder_target.append(vocab.char2index(token))
        else:
            decoder_target.append(vocab.eos_index)

    for masked_pairs in masked_indices_pairs[::-1]:
        # print(masked_pairs)
        # print(token_events[masked_pairs[0]:masked_pairs[1]])
        for pop_time in range(masked_pairs[1] - masked_pairs[0]):
            token_events.pop(masked_pairs[0])
        token_events.insert(masked_pairs[0], 'm_0')

    for token in token_events:
        tokens.append(vocab.char2index(token))

    tokens = np.array(tokens)
    decoder_target = np.array(decoder_target)

    return tokens, decoder_target, mask_track_names, mask_bar_names


def check_track_total_time(events,duration_name_to_time,duration_time_to_name,duration_times, bar_duration):


    current_time = 0
    in_duration = False
    duration_list = []
    previous_time = 0
    in_rest_s = False
    new_events = []

    if len(events) == 2:
        last_total_time_adjusted = data.time2durations(bar_duration, duration_time_to_name, duration_times)
        for token in last_total_time_adjusted[::-1]:
            events.insert(-1,token)
        events.insert(-1,'rest_e')
        return False, events

    for event in events:
        new_events.append(event)

        if in_duration and event not in vocab.duration_multi:
            total_time = data.total_duration(duration_list,duration_name_to_time)
            if in_rest_s:
                current_time = previous_time + total_time
                in_rest_s = False
            else:
                previous_time = current_time
                current_time = current_time + total_time

            in_duration = False
            if current_time >= bar_duration:
                break
            duration_list = []



        if event in vocab.duration_multi:
            in_duration = True
            duration_list.append(event)

        if event == 'rest_s':
            in_rest_s = True

    else:
        if duration_list:
            total_time = data.total_duration(duration_list, duration_name_to_time)
            if in_rest_s:
                current_time = previous_time + total_time

            else:

                current_time = current_time + total_time

   
    while new_events[-1] not in vocab.duration_multi:
        new_events.pop()
    if current_time == bar_duration:
        return True,new_events
    else:
        if current_time > bar_duration:
            difference = current_time - bar_duration
            last_total_time_adjusted = total_time - difference

        else:
            difference = bar_duration - current_time
            last_total_time_adjusted = total_time + difference

        last_duration_list = data.time2durations(last_total_time_adjusted, duration_time_to_name, duration_times)
        for _ in range(len(duration_list)):
            new_events.pop()

        new_events.extend(last_duration_list)

        return False, new_events



def restore_marked_input(src_token, generated_output):
    src_token = np.array(src_token, dtype='<U9')

    # restore with generated output
    restored_with_generated_token = src_token.copy()

    generated_output = np.array(generated_output)

    generation_mask_indices = np.where(generated_output == 'm_0')[0]

    if len(generation_mask_indices) == 1:

        mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
        generated_result_sec = generated_output[generation_mask_indices[0] + 1:]

        #         print(len(generated_result_sec))
        restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
        for token in generated_result_sec[::-1]:
            #             print(token)
            restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)


    else:

        for i in range(len(generation_mask_indices) - 1):
            #         print(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i] + 1:generation_mask_indices[i + 1]]

            #             print(len(generated_result_sec))
            #             print(mask_indices[i])
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])

            for token in generated_result_sec[::-1]:
                #                 print(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

        else:
            #         print(i)
            mask_indices = np.where(restored_with_generated_token == 'm_0')[0]
            generated_result_sec = generated_output[generation_mask_indices[i + 1] + 1:]

            #             print(len(generated_result_sec))
            restored_with_generated_token = np.delete(restored_with_generated_token, mask_indices[0])
            for token in generated_result_sec[::-1]:
                #                 print(token)
                restored_with_generated_token = np.insert(restored_with_generated_token, mask_indices[0], token)

    return restored_with_generated_token


def generation_all(model, events, device, vocab, vocab_mode,tracks_to_generate,bars_to_generate,check_total_time=True):
    result = mask_bar_and_track(events, vocab,tracks_to_generate,bars_to_generate)
    if result is None:
        return result
    src, tgt_out, mask_track_names, mask_bar_names = result


    if int(events[0][2]) == 8:
        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = data.get_note_duration_dict(
            1.5, (int(events[0][0]), int(events[0][2])))
    else:
        duration_name_to_time, duration_time_to_name, duration_times, bar_duration = data.get_note_duration_dict(
            1, (int(events[0][0]), int(events[0][2])))

    if int(events[0][0]) >= 4 and int(events[0][2]) == 4:
        no_whole_duration = False
    else:
        no_whole_duration = True


    src_masked_nums = np.sum(src == vocab.char2index('m_0'))
    tgt_inp = []
    total_generated_events = []

    if src_masked_nums == 0:
        return None
    total_corrected_times = 0
    corrected_times = 0
    with torch.no_grad():
        mask_idx = 0
        while mask_idx < src_masked_nums:

            this_tgt_inp = []
            is_time_correct = False
            this_tgt_inp.append(vocab.char2index('m_0'))
            this_generated_events = []
            this_generated_events.append('m_0')
            total_grammar_correct_times = 0
            # SMER
            if vocab_mode == 0:

                in_pitch = False
                in_rest_e = False
                in_rest_s = False
                in_continue = False

            #  REMI
            else:
                no_pitch = True
                no_step = False
                no_duration = True


            while this_tgt_inp[-1] != vocab.char2index('<eos>') and len(this_tgt_inp) < 100:


                if len(this_tgt_inp) == 1:
                    mask_track_name = 'track_' + f'{str(mask_track_names[mask_idx])}'
                    track_idx = vocab.char2index(mask_track_name)

                    this_tgt_inp.append(track_idx)
                    this_generated_events.append(mask_track_name)


                    continue

                output, weight = model_generate(model, torch.tensor(src), tgt_inp + this_tgt_inp, device,
                                                return_weights=True)
                if vocab_mode == 0:

                    if in_rest_s:

                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True,no_eos=True,no_whole_duration=True)
                        while index in vocab.rest_indices or index == vocab.eos_index or index == vocab.duration_only_indices[0]:
                            index = sampling(output[-1], vocab, no_rest=True, no_eos=True,no_whole_duration=True)

                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print("in_rest_s failed")
                                break

                        event = vocab.index2char(index)

                    elif in_continue:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_duration=True,no_continue=True,no_eos=True)
                        while index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_duration=True, no_continue=True,
                                             no_eos=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('in continue failed')
                                break

                        event = vocab.index2char(index)

                    elif in_pitch:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_rest=True, no_continue=True,
                                         no_whole_duration=no_whole_duration,no_eos=True)
                        while index not in vocab.duration_only_indices and index not in vocab.pitch_indices:
                            index = sampling(output[-1], vocab, no_rest=True, no_continue=True,
                                             no_whole_duration=no_whole_duration, no_eos=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('in pitch failed')
                                break
                        event = vocab.index2char(index)



                    elif in_rest_e:
                        sampling_times = 0

                        index = sampling(output[-1], vocab, no_pitch=True, no_rest=True,no_continue=True,
                                         no_whole_duration=no_whole_duration,no_eos=True)
                        while index not in vocab.duration_only_indices:
                            index = sampling(output[-1], vocab, no_pitch=True, no_rest=True, no_continue=True,
                                             no_whole_duration=no_whole_duration,no_eos=True)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('in rest_e failed')
                                break
                        event = vocab.index2char(index)


                    elif len(this_tgt_inp) == 2:
                        index = sampling(output[-1], vocab, no_duration=True)
                        sampling_times = 0
                        while index in vocab.duration_only_indices:
                            index = sampling(output[-1], vocab,no_duration=True)

                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('start failed')
                                break

                        event = vocab.index2char(index)

                    else:
                        # free state
                        index = sampling(output[-1], vocab, no_whole_duration=no_whole_duration)

                        event = vocab.index2char(index)

                    if index == vocab.continue_index:
                        in_continue = True
                        in_rest_s = False


                    if index in vocab.pitch_indices:
                        in_pitch = True
                        in_rest_s = False
                        in_continue = False


                    if index in vocab.duration_only_indices:
                        in_rest_e = False
                        in_pitch = False

                    if event == 'rest_s':
                        in_rest_s = True

                    if event == 'rest_e':
                        in_rest_e = True

                    if event == '<eos>':
                        if check_total_time:
                            is_time_correct, this_generated_events = check_track_total_time(this_generated_events,
                                                                                            duration_name_to_time,
                                                                                            duration_time_to_name,
                                                                                            duration_times,
                                                                                            bar_duration)
                        else:
                            is_time_correct = True



                else:
                    # step or eos
                    if no_pitch and no_duration:
                        index = sampling_step_single(output[-1], vocab, no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                        sampling_times = 0
                        # # step
                        while index not in vocab.step_indices and index != vocab.eos_index:
                            index = sampling_step_single(output[-1], vocab,  no_pitch=no_pitch,no_step=no_step,no_duration=no_duration)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('empty track here')
                                break

                        event = vocab.index2char(index)


                        no_pitch = False
                        no_duration = True
                        no_step = True

                    # pitch
                    elif no_step and no_duration:

                        index = sampling_step_single(output[-1], vocab, no_step=no_step,
                                                     no_duration=no_duration)
                        sampling_times = 0
                        while index not in vocab.pitch_indices:
                            index = sampling_step_single(output[-1], vocab, no_step=no_step, no_duration=no_duration)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('pitch failed here')
                                break
                        event = vocab.index2char(index)

                        no_duration = False
                        no_step = True
                        # if event != this_generated_events[-1]:
                        #     no_duration = False
                        #     no_step = True
                        # else:
                        #     continue


                    elif no_step:

                        index = sampling_step_single(output[-1], vocab, no_step=no_step)
                        sampling_times = 0
                        while index in vocab.step_indices:
                            index = sampling_step_single(output[-1], vocab,  no_step=no_step)
                            sampling_times += 1
                            total_grammar_correct_times += 1
                            if sampling_times > 10:
                                print('step failed here')
                                break
                        event = vocab.index2char(index)
                        if index in vocab.duration_only_indices:

                            no_pitch = True
                            no_duration = True
                            no_step = False
                    else:
                        pass

                this_tgt_inp.append(index)
                this_generated_events.append(event)

            if vocab_mode == 1:
                mask_idx += 1
                tgt_inp.extend(this_tgt_inp[:-1])
                total_generated_events.extend(this_generated_events[:-1])

            else:
                if is_time_correct:
                    if corrected_times > 5:
                        print(f'iterated times is {corrected_times}')
                    mask_idx += 1
                    tgt_inp.extend(this_tgt_inp[:-1])
                    total_generated_events.extend(this_generated_events[:-1])
                    total_corrected_times += corrected_times
                    corrected_times = 0
                else:
                    corrected_times += 1
                    if corrected_times > 10:
                        print(f'corrected times > 10, continue generation')
                        mask_idx += 1
                        tgt_inp.extend(this_tgt_inp[:-1])
                        total_generated_events.extend(this_generated_events[:-1])
                        total_corrected_times += corrected_times
                        corrected_times = 0


    src_token = []
    if vocab_mode == 0:
        if total_corrected_times > 0:
            print(f'total time corrected times is {total_corrected_times}')

    for i, token_idx in enumerate(src):
        src_token.append(vocab.index2char(token_idx.item()))

    tgt_output_events = []
    for i, token_idx in enumerate(tgt_out):
        if token_idx in vocab.structure_indices[1:]:
            tgt_output_events.append('m_0')
        if token_idx != vocab.char2index('<eos>'):
            tgt_output_events.append(vocab.index2char(token_idx.item()))

    return restore_marked_input(src_token, total_generated_events),restore_marked_input(src_token, tgt_output_events), mask_track_names, mask_bar_names

