from training import run, load_model_and_only_test

while True:
    switch = input("Do you want to train or test the model only? (train/test/exit): ")

    if switch == "train":
        switch = input("Do you want to load the model and resume training or start training from scratch? (load/scratch): ")
        if switch == "load":
            epoch = int(input("Enter the epoch number to load the model from: "))
            run(resume_train=True, start_epoch=epoch)
        elif switch == "scratch":
            run(resume_train=False)
        else:
            print("Invalid input. Please enter 'load' or 'scratch'\n")
        
    elif switch == "test":
        epoch = int(input("Enter the epoch number to load the model from: "))
        load_model_and_only_test(epoch)

    elif switch == "exit":
        break

    else:
        print("Invalid input. Please enter 'resume', 'test' or 'exit'\n")