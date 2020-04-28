import time
import joblib
import os
import cv2
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F 
#from fireup import EpochLogger
from utils.logx import EpochLogger

def render_frame(env, length, ret, primitive_name, render, record=False, caption_off=False):
    if not render and not record:
        return None
    if record:
        raw_img = env.unwrapped.render_frame()
        raw_img = np.asarray(raw_img, dtype=np.uint8).copy()
    
        # write info
        if not caption_off:
            text = ['{:4d} {:.2f} {}'.format(length, ret, primitive_name)]
            x0, y0, dy = 10, 50, 50
            for i, t in enumerate(text):
                cv2.putText(raw_img, t, (x0, y0+i*dy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 128, 128), 2, cv2.LINE_AA)

    if render:
        env.render()
        time.sleep(0.03)
#        raw_img = cv2.resize(raw_img, (500, 500))
##        cv2.imshow(env.spec.id, raw_img)
#        cv2.imshow("env.spec.id", raw_img)
#        cv2.waitKey(1)
    return raw_img if record else None


def load_policy(fpath, itr='last'):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x.split('.')[1]) for x in os.listdir(fpath) if 'ActorCritic.' in x]
        itr = '.%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '.%d'%itr

    # load the things!
    model = torch.load(osp.join(fpath, 'ActorCritic'+itr+'.pt'))
    model.eval()

    # get the model's policy
    get_action = model.policy

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, itr


def run_policy(env, get_action, ckpt_num, max_con, con, max_ep_len=100, num_episodes=100, 
               fpath = None, render=False, record=True, video_caption_off=False):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    output_dir = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),"log/tmp/experiments/%i"%int(time.time()))
    logger = EpochLogger(output_dir = output_dir)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    visual_obs = []
    c_onehot = F.one_hot(torch.tensor(con), max_con).squeeze().float()
    while n < num_episodes:
        
        vob = render_frame(
            env, ep_len, ep_ret, 'AC', render, record, caption_off=video_caption_off)
        visual_obs.append(vob)
        concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
        a = get_action(concat_obs)
        o, r, d, _ = env.step(a[0].detach().numpy()[0])
        ep_ret += r
        ep_len += 1
        d = False
        
        if d or (ep_len == max_ep_len):
            vob = render_frame( env, ep_len, ep_ret, 'AC', render, record, 
                               caption_off=video_caption_off)
            visual_obs.append(vob)  # add last frame
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
            
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    if record:
        # temp_info: [video_prefix, ckpt_num, ep_ret, ep_len, con]
        temp_info = ['', ckpt_num, ep_ret, ep_len, con]
        logger.save_video(visual_obs, temp_info, fpath)


def run(args):
    env, get_action, ckpt_num = load_policy(args.fpath,args.itr if args.itr >=0 else 'last')
    
#    import gym
#    env = gym.make("HalfCheetah-v2")
    assert args.con in [-1]+list(range(args.max_con))
    
    if args.con > -1:
        run_policy(env, get_action, ckpt_num, args.max_con, args.con, args.len, args.episodes, args.fpath, 
                   args.render, args.record, not(args.not_caption))
        return
    for con in range(args.max_con):
        run_policy(env, get_action, ckpt_num, args.max_con, con, args.len, args.episodes, args.fpath, 
                   args.render, args.record, not(args.not_caption))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=250)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--max_con', '-maxc', type=int, default=10)
    parser.add_argument('--con', '-c', type=int, default=-1)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--record', '-v', action='store_true')
    parser.add_argument('--not_caption', '-nc', action='store_true')
    
    args = parser.parse_args()
    run(args)


'''
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-390/libGL.so
'''