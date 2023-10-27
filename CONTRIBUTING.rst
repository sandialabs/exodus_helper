.. contents:: *Table of Contents*

Ways to Contribute
------------------

Individual or groups are very welcome to contribute to |project|.  There are four main ways of contributing to the |project| project:

 (1) Submitting issues related to bugs or desired enhancements.
 (2) Adding new or improved functionality to the existing codebase.
 (3) Fixing outstanding bugs with the existing codebase.
 (4) Contributing or improving the documentation (`docs`).


Submitting Issues
-----------------

The best way to submit issues is via |GithubIssue|_.  Please verify that your issue is not being currently addressed by other issues or pull requests by using the Github search tool to look for key words in the project issue tracker.  If you find a matching issue, feel free to comment in the chat to indicate you think that issue is important and you might have ideas on how to fix it - your use case/experience may give you insight on how to solve the problem that is not currently being thought of - a collaborative solution is ideal!

Addressing Issues
-----------------

If you would like to address an issue directly, users are strongly encouraged to submit patches for new or existing issues via pull requests.  This is especially appreciated for simple fixes like typos or tweaks to documentation.

Contributors are also encouraged to contribute new code to enhance |project|'s functionality, also via pull requests. Please consult the |project| |Documentation|_ to ensure that any new contribution does not strongly overlap with existing functionality.

The recommended workflow for contributing to |project| is outlined in the :ref:`DeveloperGuidelines`.

.. _DeveloperGuidelines:

Developer Guidelines
--------------------

Getting Started
^^^^^^^^^^^^^^^

- Clone the |project| repo to your local disk and navigate to the created folder:

::

   $ git clone git@github.com:sandialabs/exodus_helper.git
   $ cd exodus_helper

- To install |project| such that your local Python installation will reference the development version you are working on run:

::

   $ pip install -e .

This creates a link in your `site-packages` distribution that points to the developing source code.

Developer workflow
^^^^^^^^^^^^^^^^^^

The development workflow should for the most part follow this structure.

- The latest stable release of the code will be in the `stable` branch.  Let's assume version 1.1.0 is the latest stable version.
- All development will be done with respect to a development branch until ready to release the next stable version.  Let's assume that the current development branch is `dev-v1.2.0`.  On this branch, you should find that the version attribute is set to `1.2.0beta` or `1.2.0alpha` (the "beta" or "alpha" will be removed when ready for release).

Suppose you want to address a new feature (e.g., issue #7) that is scheduled to be included in the release of version 1.2.0.  You would follow these steps to address that issue:

(1) From your computer, checkout the latest version of the development branch and pull the latest changes.

::

   $ git checkout dev-v1.2.0
   $ git pull

(2) Create a new branch where you can safely address the feature.  As it was issue #7, a descriptive branch name like `feat-7-new-method` is a good idea.

::

   $ git checkout -b feat-7-new-method

(3) Modify the existing files and/or add new ones to address the feature.

::

   $ git add <modified_file(s)> <new_file(s)>
   $ git commit

Write a descriptive commit message such as "Closes #7 - added feature to do amazing things".  Note, you can commit incremental changes to the code(s).  You do not have to get everything figured out before you start committing files!

(4) If you have completed the task, or you simply want to save your progress to the remote server, then push your local branch to Github

::

   $ git push -u origin feat-7-new-method

For subsequent pushes to the remote branch, you simply need to run `git push`.

(5) Once the task is complete, you should submit a |PullRequest|_ in order to merge your feature branch into the development branch.  As part of the request, please provide a title, labels, and request reviewers.

- Source: `feat-7-new-method` | Target: `dev-v1.2.0`
- Title: e.g., `Closes #7 - added feature to do something amazing`
- Labels: e.g., `Enhancement` or `Documentation`
- Reviewers: Select from drop-down list

See :ref:`PullRequestChecklist` for more details on what steps should be taken before submitting a pull request.

(6) Once the pull request is submitted, it will run the continuous-integration (CI) system to make sure all tests still run and documentation is built successfully.  The requested reviewers will receive an email from Github and they will review your request.  There may be requests from the reviewers that you make additional changes to the code, which can be done by committing more changes from your local branch and pushing them to the remote (the pull request will update to reflect these changes).  Once your submission has passed all tests, reviewer requirements, and all merge conflicts have been resolved, then the merge can be finished.

(7) Once the merge is complete, go back to your local version of the development branch and pull the changes.

::

   $ git checkout dev-v1.2.0
   $ git pull

(8) Finally, you can delete your local version of the feature branch
::

   $ git branch -d feat-7-new-method

It is a good idea to clean up the feature/patch branches as soon as you are done with the task so that you don't later accidentally work from the wrong branch.

.. _PullRequestChecklist:

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  All public methods must have informative docstrings with sample usage when appropriate.

*  Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to

  (1) indicate you are working on something to avoid duplicated work,
  (2) request broad review of functionality or API, or
  (3) seek collaborators.

*  All other tests pass when everything is rebuilt from scratch.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Run any of the pre-existing |Examples|_ that contain analyses that would be affected by your changes to ensure that nothing breaks. This is a useful opportunity to not only check your work for bugs that might not be revealed by unit testing, but also to show how your contribution improves |project| for end users.


You can also check for common programming errors with the following
tools:

* Check code **coverage** (at least 80%) with:

::

  $ pip install pytest pytest-cov
  $ py.test -ra --cov=exodus_helper --cov-report=term-missing

Your test methods need to start with the prefix "test\_" to be run in `pytest`.

* (Optional) Check code style (no `flake8` warnings) with:

::

  $ pip install flake8
  $ flake8 exodus_helper/path_to_module.py

Development Cycle
-----------------
New versions of the code will periodically be released.  The frequency of the releases will vary based on the demand for the code.  The code follows a versioning structure of <major>.<feature>.<bug>, so several bug releases may come out between feature releases if there is not much demand for new functionality.  So long as the major release number is the same, the developers believe the code to be backwards compatible.

A |MileStones|_ will be created for each expected release, and issues will be assigned based on priority.  Once all issues for a milestone have been addressed, then it will be time to release the version associated with that milestone.  This will be accomplished by merging the development branch into the `stable` branch via a pull request (test, review, etc.).  Once complete, we can go to our local version of the `stable` branch and tag a version (you can also do this directly on Github).

::

   $ git checkout stable
   $ git pull

Version number should be `1.2.0`.  We can now tag it locally and push to Github.

::

   $ git tag v1.2.0
   $ git push origin v1.2.0

From Github, we can create a |Releases|_, which is where users can download their desired version of the source code.

The workflow described above essentially requires all code development to be driven by specific tasks in the |GithubIssue|_.  Using this type of workflow helps to enforce small, easily trackable changes to the code.  Furthermore, by referencing an issue ID number (#<issue id>) in a commit message it is also useful because the commit becomes part of the conversation history for that issue.  The use of the keyword "Closes #<issue id>" in a commit message or pull request will cause the issue to automatically close once the code is merged back into the main branch.  However, you can manually close the issue after merging to a development branch.


.. |project| replace:: exodus_helper

.. |GithubIssue| replace:: Github Issue Tracker
.. _GithubIssue: https://github.com/sandialabs/exodus_helper/issues

.. |Documentation| replace:: documentation
.. _Documentation: https://exodus-helper.readthedocs.io/en/latest/index.html

.. |Repository| replace:: repository
.. _Repository: https://github.com/sandialabs/exodus_helper

.. |Examples| replace:: examples
.. _Examples: https://github.com/sandialabs/exodus_helper/tree/main/examples

.. |MileStones| replace:: milestone
.. _MileStones: https://github.com/sandialabs/exodus_helper/milestones

.. |Releases| replace:: release
.. _Releases: https://github.com/sandialabs/exodus_helper/releases

.. |PullRequest| replace:: pull request
.. _PullRequest: https://github.com/sandialabs/exodus_helper/pulls
