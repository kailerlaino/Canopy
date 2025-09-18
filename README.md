# Canopy
This repository contains code and documentation to train and evaluate `Canopy` models. The easiest way of reproducing our results would be to use CloudLab along with our [custom CloudLab profile + image](https://www.cloudlab.us/p/VerifiedMLSys/orca). 

## Cloudlab setup (training + eval)
1. Create a new job on `CloudLab` with 17 nodes (if you want to train your own model) or with just 1 node (if you want to evaluate our pre-existing checkpoints). When [creating the job](https://www.cloudlab.us/instantiate.php), make sure to click the "change profile" button and search for a profile named "Orca" (https://www.cloudlab.us/p/VerifiedMLSys/orca).

2. Once the job is created `ssh` into the nodes, and verify that linux kernel image being used is `4.13.1-*-learner` to confirm that you used the right profile. 


3. [from your desktop] Setup the cloudlab nodes, by running the next command. When you run this command it'll prompt you for a location for an RSA - let it be the default location and do not add a passphrase.
```bash
# if you created a job with 17 nodes
./cloudlab/config.sh < cloudlab-experiment-name > 0 16 wisc < cloudlab-username > < cloudlab-projectname >

# if you created a job with just one node
./cloudlab/config.sh < cloudlab-experiment-name > 0 0 wisc < cloudlab-username > < cloudlab-projectname >
```

4. [from your desktop] Run `./cloudlab/copy_files.sh` - the previous command should have printed out the exact arguments you should use for this script. This will create a directory called `~/ConstrainedOrca` on all the nodes and copy the required code over to the machines (from your machine).

5. [from your desktop] Run this command to get a list of IPs. We will use `node0` of the cloudlab job you created for the learner and nodes 1 to 16 for the actors. This command populates the list of IP addresses of actor nodes:

```bash
./cloudlab/get_ips.sh <cloudlab-experiment-name> 1 16 wisc <cloudlab-username> <cloudlab-projectname> # if you created 17 nodes
./cloudlab/get_ips.sh <cloudlab-experiment-name> 0 0 wisc <cloudlab-username> <cloudlab-projectname> # if you created 1 node
```

5. [from node0 of the cloudlab job] `ssh` into `node0` of your job, go into the directory called `~/ConstrainedOrca` and then run `./cloudlab/setup_params.sh "<list of ips>" 16` where `<list of ips>` is the space separated list you got in the previous step.

6. [from node0] Run the following command to download the traces and the checkpoints onto all cloudlab nodes. In this script, our checkpoints are being downloaded from a [Zenodo archive](https://zenodo.org/records/17145110/files/canopy_artefact.zip), where we have made them public:
```bash
./cloudlab/download_traces_ckpt.sh
```

## Training a Canopy model
1. If you wish to only evaluate our (final) trained checkpoints (i.e., you've created only one node), please skip this section.
2. [from node0 of your cloudlab job] `v9_multi_train.sh` is the entrypoint to training. Run ONE of the following commands (in tmux/screen) to train a deep buffer or shallow buffer model respectively (might take a couple hours): 

```bash
bash scripts/v9_multi_train.sh 0.25 5 3 raw-sym 25 0 11 25 $USER # train deep buffer model 
bash scripts/v9_multi_train.sh 0.25 5 3 raw-sym 25 0 12 25 $USER # train shallow buffer Canopy model
```

3. You should start seeing logging that looks like the following.
```bash
[TRAIN] Started actor 3 on <ip>:<port> (MM port: <port>) with PID <pid> [delay=10; bw=24; bdp=40; qs=200]
```

3. From `node0`, you can ssh into these nodes (using IPs from logs / cloudlab SSH addresses) to look at `~/actor_logs/` to see the `stdout` (and `stderr`) of each actor. Also, `~/ConstrainedOrca/rl-module/training_log` contains training logs for each actor in a separate file on all the actor nodes.

4. [from `node0` on cloudlab] Once training is done, run `scripts/collate_train_files.sh` (on node0), which will collect all stdout/stderr messages from all actors as well as train logs (i.e. loss) and checkpoints and put it into a folder `~/results` on `node0`

5. Within this `~/results/` dir, the main file that "matters" is the `learner_ckpt.tar.gz`. Extracting this should result in a file structure that looks something like this:
```
seed0/
└── learner0-v9_actorNum256_multi_lambda0.25_ksymbolic5_k3_raw-sym_threshold25_seed0_constraints_id11_xtwo25
    ├── checkpoint
    ├── events.out.tfevents.1738135603.node0.c3.verifiedmlsys-pg0.wisc.cloudlab.us
    ├── graph.pbtxt
    ├── model.ckpt-291373.data-00000-of-00001
    ├── model.ckpt-291373.index
    ├── model.ckpt-291373.meta
    ├── model.ckpt-293121.data-00000-of-00001
    ├── model.ckpt-293121.index
    ├── model.ckpt-293121.meta
    ├── model.ckpt-294884.data-00000-of-00001
    ├── model.ckpt-294884.index
    ├── model.ckpt-294884.meta
    ├── model.ckpt-296642.data-00000-of-00001
    ├── model.ckpt-296642.index
    ├── model.ckpt-296642.meta
    ├── model.ckpt-298398.data-00000-of-00001
    └── model.ckpt-298398.index
```

## Eval
1. [pretrained checkpoint] Go into `~/checkpoints` on `node0` and extract one of the `tar.gz` files in there. You will get a very similar directory structure as shown above in the last step of train instructions. 
2. Move the `learner0*` checkpoint directory to be inside `~/ConstrainedOrca/rl-module/train_dir/seed0/`. This `learner0*` folder name is the `<model_name>`.
3. Run `./scripts/eval_orca.sh <model_name> <trace_dir> <results_dir> <start_run> <end_run> <constraints_id>`. `<start_run>` and `<end_run>` let you set how many times you want to repeat the experiment. e.g., setting `<start_run> <end_run>` to `1 5` will run each trace 5 times, independently.
3. `<trace_dir>` should be one of `~/traces/sage_traces`, `~/traces/synthetic`, or `~/traces/variable-links`.
4. `<results_dir>` is where you want evaluation results saved. This `eval_orca.sh` script will (one by one) run all traces inside the `<trace_dir>` you provided with the `<model_name>` you provided.

## Plotting and analysis
0. Use `cd scripts && ./process_down_file.sh <result_dir>` to preprocess down files.
1. Use `./scripts/plots/plot_thr.py` for motivation figures
2. Use `./scripts/plots/plot_thr_delay.py` for thr vs delay plots.