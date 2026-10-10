# DVC Dataset Push Instructions

The dataset was tracked locally but was not pushed to the DagsHub DVC remote, causing `dvc pull` to fail for external contributors and notebooks. Follow these instructions to securely push the missing `dataset` directory to the remote without leaking credentials into Git.

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

If successful, repeating `dvc status -c` should return a message indicating everything is synchronized. A correct result looks like:
```
Data and pipelines are up to date.
```

## 3. Verify in a Fresh Clone

To guarantee the fix works for users, verify the pull in a fresh clone of the repository:

```bash
git clone https://github.com/Bhar07gavi/Cloud_Removal.git cloud-removal-test
cd cloud-removal-test
dvc pull -r dagshub
```

If the pull completes without the `Missing cache files` error and populates the `dataset/` directory, the issue is fully resolved!
