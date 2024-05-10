
1. To connect to ssh: (see sftp doc for file transfer)
```commandline
ssh -X cslogin@ssh.ccv.brown.edu
```

2. To apply for gpu:

```commandline
interact -q gpu -g 1 -f ampere -m 20g -n 4 -t 15:00:00
```

3. load CUDA
```commandline
module load cuda/12.2.0-4lgnkrh
module load cudnn/8.9.6.50-12-56zgdoa
```

4. Activate env

```commandline
source ../../pytorch.venv/bin/activate
```

5. Run scripts
```commandline
bash scripts/run_train.sh
```