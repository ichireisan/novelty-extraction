# Publish this repository on GitHub

The source is designed for a public MIT-licensed repository. Run the tests and inspect the staged diff before publishing. Local databases, generated run artifacts, credentials, and the original conversation are ignored by default.

## Authenticate

```bash
gh auth login --hostname github.com
gh auth status
```

Use your own GitHub account or an organization where you have repository-creation permission. Do not paste access tokens into issues, commits, or chat.

## Initialize and commit, if needed

If this directory already has a Git repository and an initial commit, skip these steps.

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Add selective learning research prototype"
```

Git may ask you to configure your author name and email. Use the identity you want attached to public commits. Do not overwrite an existing remote if the directory belongs to another project.

## Create the public repository

```bash
gh repo create novelty-extraction \
  --public \
  --description "Experimental budgeted selective learning for AI agents" \
  --source . \
  --remote origin \
  --push
```

Replace `novelty-extraction` with `YOUR-ORG/novelty-extraction` when publishing under an organization. If the name already exists, choose another name or connect the intended existing repository explicitly.

## After publication

- Confirm the Actions workflow passes.
- Add the repository URL to `CITATION.cff` if desired.
- Enable Discussions if you want a place for research proposals.
- Create a `v0.1.0` release when you are ready to mark the initial prototype.

The included reference report is synthetic. Keep that description when announcing results.
