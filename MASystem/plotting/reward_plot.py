from matplotlib import pyplot as plt

def plot_acc_reward(reward_list,plot_name):
    plt.clf()
    plt.plot(reward_list)
    plt.xlabel("episode")
    plt.ylabel("acc_reward_per_episode")
    plt.savefig(plot_name+'.png')


def plot_acc_reward_per_window(reward_list, window_length, plot_name):
    windowed_list = get_windowed_list(reward_list, window_length)
    plt.clf()
    plt.plot(windowed_list)
    plt.xlabel("episode")
    plt.ylabel("acc_reward_per_window")
    plt.savefig(plot_name+'.png')


def get_windowed_list(reward_list, window_length):
    import statistics
    main_list = []
    list_length = len(reward_list)
    r = int(window_length/2)
    for i in range(r, list_length-r):
        temp_list = reward_list[i-r:i+r:]
        m = statistics.mean(temp_list)
        main_list.append(m)
    return main_list


