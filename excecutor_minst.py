from training import run, load_model_and_only_test

while True:
    switch = input("[MINST] Do you want to train or test the model only? (train/test/exit): ")

    if switch == "train":
        switch = input("[MINST] Do you want to load the model and resume training or start training from scratch? (load/scratch): ")
        if switch == "load":
            epoch = int(input("[MINST] Enter the epoch number to load the model from: "))
            run(resume_train=True, start_epoch=epoch, robotics_task=False)
        elif switch == "scratch":
            run(resume_train=False, robotics_task=False)
        else:
            print("[MINST] Invalid input. Please enter 'load' or 'scratch'\n")
        
    elif switch == "test":
        epoch = int(input("[MINST] Enter the epoch number to load the model from: "))
        load_model_and_only_test(epoch)

    elif switch == "exit":
        break

    else:
        print("[MINST] Invalid input. Please enter 'resume', 'test' or 'exit'\n")