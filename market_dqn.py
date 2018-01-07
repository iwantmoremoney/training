import numpy as np

from market_env import MarketEnv
from market_model_builder import MarketModelBuilder
from datetime import datetime
import os
import sys

class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        inputs = []

        dim = len(self.memory[0][0][0])
        for i in xrange(dim):
            inputs.append([])

        targets = np.zeros((min(len_memory, batch_size), num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            for j in xrange(dim):
                inputs[j].append(state_t[j][0])

            #inputs.append(state_t)
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        
        #inputs = np.array(inputs)
        inputs = [np.array(inputs[i]) for i in xrange(dim)]

        return inputs, targets

if __name__ == "__main__":
    import sys
    import codecs

    dataset = sys.argv[1]
    env = MarketEnv( dataset=dataset)

    # parameters
    epsilon = .5  # exploration
    min_epsilon = 0.1
    epoch = 100000
    max_memory = 5000
    batch_size = 128
    discount = 0.8

    from keras.optimizers import SGD
    m = MarketModelBuilder()
    model = m.getModel()
    sgd = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss='mse', optimizer='rmsprop')

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory = max_memory, discount = discount)

    print "[INFO][VER][DATA] {} {}".format( dataset, os.popen('git ls-tree master data').read().strip() ) 
    print "[INFO][VER][ENV] {} {}".format( dataset, os.popen('git show | head -n 1').read().strip() ) 
    print "[INFO][VER][MODEL] {}".format( m.name() )

    if not os.path.exists("reports"):
        print '[ERROR] No reports folder.  Please submodule init and update'
        sys.exit()
    if not os.path.exists("reports/{}".format(m.name())):
        os.mkdir("reports/{}".format(m.name()))
    file_model = "reports/{}/model".format(m.name())

    if os.path.exists(file_model):
        print '[WARNING] {} model existed.  Overwrite it.'.format( m.name() )

    # Train
    win_cnt = 0
    for e in range(epoch):
        start_time = datetime.now()
        loss = 0.
        game_over = False
        # get initial input
        input_t = env.reset()
        cumReward = 0

        while not game_over:
            input_tm1 = input_t
            isRandom = False

            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, env.action_space.n, size=1)[0]

                isRandom = True
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

                #print "  ".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, q[0].tolist())])
                if np.nan in q:
                    print "OCCUR NaN!!!"
                    exit()

            # apply action, get rewards and new state
            input_t, reward, game_over, info = env.step(action)
            #print input_t[0]
            cumReward += reward

            if info['action'] in ( 'HOLD' ):
                print "%08s  %15s  %05.2f  %010f" % ( info['date'], "", info['close'], info['balance'][2] )
            else:
                print "%08s  %15s  %05.2f  %010f" % ( info['date'], info['action'], info['close'], info['balance'][2] )
            print("[INFO][BALANCE] {}".format( info['balance'] ) )

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

        if info['ratio'] > 100 and info['status'] == 'FINISH' and game_over:
            win_cnt += 1

        print("[INFO][EPOCH] Epoch {:03d}/{} | Total: {:10.2f} | Ratio: {} | Status: {} | Loss {:.4f} | Reward {:.4f} | Win count {} | Epsilon {:.4f} | Time: {}".format(e, epoch, info['balance'][2], info['ratio'], info['status'], loss, cumReward, win_cnt, epsilon, datetime.now() - start_time ))
        print("[INFO][POSITION] {}".format( info['position'] ) )
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save_weights( file_model, overwrite=True)
        epsilon = max(min_epsilon, epsilon * 0.99)
