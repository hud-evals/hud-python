"""Tests for git utilities."""

from __future__ import annotations

from unittest import mock

from hud.cli.utils.git import get_git_info, get_git_remote_url, normalize_github_url


class TestNormalizeGithubUrl:
    """Test GitHub URL normalization."""

    def test_normalize_ssh_url(self):
        """Test normalizing SSH format URL."""
        url = "git@github.com:user/repo.git"
        result = normalize_github_url(url)
        assert result == "https://github.com/user/repo"

    def test_normalize_https_with_git_suffix(self):
        """Test normalizing HTTPS URL with .git suffix."""
        url = "https://github.com/user/repo.git"
        result = normalize_github_url(url)
        assert result == "https://github.com/user/repo"

    def test_normalize_git_protocol(self):
        """Test normalizing git:// protocol URL."""
        url = "git://github.com/user/repo.git"
        result = normalize_github_url(url)
        assert result == "https://github.com/user/repo"

    def test_normalize_already_clean(self):
        """Test URL that's already normalized."""
        url = "https://github.com/user/repo"
        result = normalize_github_url(url)
        assert result == "https://github.com/user/repo"

    def test_normalize_with_github_com_colon(self):
        """Test URL with github.com: format."""
        url = "ssh://github.com:user/repo.git"
        result = normalize_github_url(url)
        assert result == "https://github.com/user/repo"


class TestGetGitRemoteUrl:
    """Test getting git remote URL."""

    @mock.patch("subprocess.run")
    def test_get_remote_url_success(self, mock_run):
        """Test successfully getting remote URL."""
        # First call checks if we're in a git repo
        mock_run.side_effect = [
            mock.Mock(returncode=0),  # git rev-parse --git-dir
            mock.Mock(returncode=0, stdout="git@github.com:user/repo.git\n"),  # git config
        ]

        result = get_git_remote_url()
        assert result == "https://github.com/user/repo"

    @mock.patch("subprocess.run")
    def test_get_remote_url_not_git_repo(self, mock_run):
        """Test when not in a git repository."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(128, "git")

        result = get_git_remote_url()
        assert result is None

    @mock.patch("subprocess.run")
    def test_get_remote_url_no_remote(self, mock_run):
        """Test when no remote origin exists."""
        from subprocess import CalledProcessError

        mock_run.side_effect = [
            mock.Mock(returncode=0),  # git rev-parse --git-dir
            CalledProcessError(1, "git"),  # git config fails
        ]

        result = get_git_remote_url()
        assert result is None

    @mock.patch("subprocess.run")
    def test_get_remote_url_empty(self, mock_run):
        """Test when remote URL is empty."""
        mock_run.side_effect = [
            mock.Mock(returncode=0),
            mock.Mock(returncode=0, stdout=""),
        ]

        result = get_git_remote_url()
        assert result is None


class TestGetGitInfo:
    """Test getting comprehensive git info."""

    @mock.patch("hud.cli.utils.git.get_git_remote_url")
    @mock.patch("subprocess.run")
    def test_get_git_info_success(self, mock_run, mock_get_url):
        """Test successfully getting all git info."""
        mock_get_url.return_value = "https://github.com/user/repo"
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="main\n"),  # branch
            mock.Mock(returncode=0, stdout="abc1234\n"),  # commit
        ]

        result = get_git_info()

        assert result["remote_url"] == "https://github.com/user/repo"
        assert result["branch"] == "main"
        assert result["commit"] == "abc1234"

    @mock.patch("hud.cli.utils.git.get_git_remote_url")
    @mock.patch("subprocess.run")
    def test_get_git_info_no_remote(self, mock_run, mock_get_url):
        """Test git info when no remote exists."""
        mock_get_url.return_value = None
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="feature-branch\n"),
            mock.Mock(returncode=0, stdout="def5678\n"),
        ]

        result = get_git_info()

        assert result["remote_url"] is None
        assert result["branch"] == "feature-branch"
        assert result["commit"] == "def5678"

    @mock.patch("hud.cli.utils.git.get_git_remote_url")
    @mock.patch("subprocess.run")
    def test_get_git_info_subprocess_error(self, mock_run, mock_get_url):
        """Test git info when subprocess fails."""
        from subprocess import CalledProcessError

        mock_get_url.return_value = "https://github.com/user/repo"
        mock_run.side_effect = CalledProcessError(1, "git")

        result = get_git_info()

        assert result["remote_url"] == "https://github.com/user/repo"
        assert "branch" not in result
        assert "commit" not in result
