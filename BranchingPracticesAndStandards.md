# Branching Practices and Standards for MiadTechnologies

## Overview

In this repository, we use a structured branching strategy to keep our code organized, stable, and easily manageable. Our branching workflow consists of **two main branches**—`main` and `dev`—and **feature branches** created for specific work. This approach helps to streamline the development process, manage updates efficiently, and maintain code quality.

---

## Branch Structure

### 1. `main` Branch

- The `main` branch is the **production-ready** branch. Only stable and tested code should be merged here.
- All code on `main` is reviewed and approved before merging.
- Only authorized personnel (project maintainers) can merge code into `main`.

### 2. `dev` Branch

- The `dev` branch is for **testing** new features, bug fixes, and other updates before they go to `main`.
- Code pushed to `dev` should come through **pull requests** from feature branches and is reviewed before merging.
- All developers push their work to `dev` via feature branches.

### 3. Ad-hoc Branches

- Each new feature, bug fix, or task gets its **own branch off `dev`**.
- Ad-hoc branches should be named according to the work they address, and/or the dir or file name or path (e.g., `DataCollection-NewScraper`, `bugfix-button`, or `docs-readme`).
- Once completed, the feature branch is **merged back into `dev`** through a pull request, then reviewed, and tested. 
- Ad-hoc branches can be nested, and should be when doing work on complex code sets. For instance, when doing work in dev, and working on DataCollection, you should branch into DataCollection to do all the work specific to that functionality. Knowing that DataCollection can be a complex branch in of itself, one can further branch into a properly named branch to match the work they address (e.g `git check dev` (Main Development Branch), then `git checkout DataCollection dev` (Known branch off dev), then `git checkout -b {functionality} DataCollection` (new branch to address new changes branched off DataCollection branch))

---

## Workflow for Contributors

### Step 0: If You Do Not Have The Repository - Clone The Repository

Start by cloning the repository to your local machine:

```bash
git clone git@github.com:MiadTechnologiesLCC/MidasTechnologies.git
cd MidasTechnologies
```
> Note if you're new, or not a known contributor this will not work. Contact admins 

### Step 1: Set Up Branch Tracking

Make sure you have the latest branches set up locally:

```bash
git fetch origin
```

### Step 2: Check Out `dev` for New Work

Always base new work off the `dev` branch:

```bash
git checkout dev
git pull origin dev  # Make sure your dev branch is up to date
```

---

## Working with Ad-Hoc Branches

### Step 1: Create a New Ad-Hoc Branch

For each new task or feature, create a new branch off `dev`. Name your branch descriptively:

```bash
git checkout -b ad-hoc-branch-name dev
```

Example:

```bash
git checkout -b ad-hoc-login dev
```

### Step 2: Make and Commit Changes

As you work on the feature, make commits regularly:

```bash
git add .
git commit -m "Descriptive commit message about the change"
```

### Step 3: Push the Feature Branch to GitHub

When you’re ready to share your work, push the feature branch to GitHub:

```bash
git push -u origin ad-hoc-branch-name
```

---

## Pull Request and Review Process

Once your feature branch is ready, follow these steps:

1. **Create a Pull Request (PR)**: Go to GitHub, find your branch, and create a pull request targeting `dev`.
2. **Request a Review**: Mention the team lead or project maintainer in your PR and ask for a review.
3. **Address Feedback**: Make any requested changes by committing and pushing to your feature branch.

Once your PR is approved, it can be merged into `dev`.

---

## Merging into `main`

Only the team lead or project maintainer will manage merging from `dev` to `main` once testing and reviews are complete. This ensures `main` remains stable.
The Project Maintainer currently, and author of this file is KleinPanic

---

## Git Commands Cheat Sheet

Here’s a quick reference for Git commands relevant to our workflow:

1. **Clone the repository**:

   ```bash
   git clone git@github.com:MiadTechnologiesLCC/MidasTechnologies.git
   ```

2. **Create a new branch**:

   ```bash
   git checkout -b branch-name dev
   ```

3. **Add changes**:

   ```bash
   git add .
   ```

4. **Commit changes**:

   ```bash
   git commit -m "Your message here"
   ```

5. **Push a branch to GitHub**:

   ```bash
   git push -u origin branch-name
   ```

6. **Switch branches**:

   ```bash
   git checkout branch-name
   ```

7. **Fetch updates from GitHub**:

   ```bash
   git fetch origin
   ```

8. **Pull updates from a branch**:

   ```bash
   git pull origin branch-name
   ```

9. **Delete a branch (locally)**:

   ```bash
   git branch -d branch-name
   ```

10. **Delete a branch (remotely)**:

    ```bash
    git push origin --delete branch-name
    ```

---

## Example Workflow

Here’s an example of a typical workflow for a new feature:

1. Switch to `dev` and ensure it’s up to date:

   ```bash
   git checkout dev
   git pull origin dev
   ```

2. Create a new branch:

   ```bash
   git checkout -b feature-user-profile dev
   ```

3. Work on the feature, adding and committing changes as you go.

4. Push your branch to GitHub:

   ```bash
   git push -u origin feature-user-profile
   ```

5. Create a pull request from your branch to `dev` on GitHub, request a review, and wait for approval.

---

## Summary

- `main`: Production-ready, stable code only.
- `dev`: Testing branch; pull requests from feature branches go here.
- **Feature Branches**: Branch off `dev` for specific features or bug fixes.

Following this workflow will ensure our codebase remains organized and stable while allowing for efficient development and collaboration.

> Contact Admins with questions. 
