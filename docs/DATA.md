# DVC Dataset Sync & Push Instructions

Follow these instructions to securely synchronize the tracked `dataset` directory with the DagsHub DVC remote without leaking credentials into Git.

## 1. Configure Local DVC Credentials

To authenticate with DagsHub without committing your credentials, use the `--local` flag so DVC stores them in `.dvc/config.local` (which is properly gitignored by `.dvc/.gitignore`):

```bash
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user YOUR_USERNAME
dvc remote modify --local dagshub password YOUR_TOKEN
```

## 2. Push the Data

Check the status of your local DVC cache against the remote, then push the tracked files:

```bash
dvc status -c
dvc push -r dagshub
```

If successful, repeating `dvc status -c` prints:
```
Cache and remote 'dagshub' are in sync.
```

## 3. Verify in a Fresh Clone

To verify the pull in a fresh clone of the repository, note that the `dvc remote modify --local` credential commands from Step 1 must be repeated there:

```bash
git clone https://github.com/Bhar07gavi/Cloud_Removal.git cloud-removal-test
cd cloud-removal-test
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user YOUR_USERNAME
dvc remote modify --local dagshub password YOUR_TOKEN
dvc pull -r dagshub
```

If the pull completes without errors and populates the `dataset/` directory, the data synchronization is fully verified!
