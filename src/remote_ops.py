import subprocess
import os
import sys

class RemoteTrainer:
    def __init__(self, hostname, username, ssh_key_path, remote_project_root):
        """
        Uses SHELL execution to bypass macOS Python Sandbox restrictions.
        """
        self.target = f"{username}@{hostname}"
        self.key_path = os.path.expanduser(ssh_key_path)
        self.remote_root = remote_project_root
        
        # Build the flags string once
        self.flags = f"-4 -i {self.key_path} -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    def run_command(self, command):
        """
        Runs a generic command on Linux and returns the result.
        Useful for diagnostics or checking if a file exists.
        
        Returns: (exit_code, stdout_str, stderr_str)
        """
        # Wrap the command in quotes for SSH
        full_cmd = f"ssh {self.flags} {self.target} \"{command}\""
        
        # Run it safely using subprocess
        result = subprocess.run(
            full_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def connect(self):
        """
        Verifies connection using the system shell.
        """
        print(f"🔌 Testing connection to {self.target} (Shell Mode)...")
        code, out, err = self.run_command("echo 'Connection OK'")
        
        if code == 0:
            print("✅ Connection Established.")
        else:
            print(f"❌ Connection Failed. Exit Code: {code}")
            print(f"   Error: {err}")
            raise ConnectionError(f"Could not connect to {self.target}")

    def close(self):
        # Stateless connection, nothing to close.
        pass 

    def sync_file(self, local_path, relative_remote_path):
        """
        Uploads file using 'scp' via Shell.
        """
        remote_full_path = f"{self.remote_root}/{relative_remote_path}".replace("//", "/")
        remote_dir = os.path.dirname(remote_full_path)

        # 1. Create Folder first
        self.run_command(f"mkdir -p {remote_dir}")

        # 2. Upload
        print(f"🚀 Uploading: {os.path.basename(local_path)} -> Linux")
        cmd = f"scp {self.flags} \"{local_path}\" {self.target}:{remote_full_path}"
        
        try:
            subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ Upload failed: {e}")
            raise

    def retrieve_file(self, relative_remote_path, local_destination):
        """
        Downloads file using 'scp' via Shell.
        """
        remote_full_path = f"{self.remote_root}/{relative_remote_path}".replace("//", "/")
        
        print(f"📥 Downloading: {os.path.basename(str(local_destination))}")
        
        cmd = f"scp {self.flags} {self.target}:{remote_full_path} \"{local_destination}\""
        
        try:
            subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Remote file not found or download failed.")
            return False

    def run_training_job(self, workspace, iteration, python_bin):
        """
        Runs training and streams output.
        """
        env_setup = f"export CAMPAIGN_TAG='{workspace.campaign_tag}'; export LLM_MODEL='{workspace.raw_model_name}';"
        remote_cmd = f"{env_setup} cd {self.remote_root} && {python_bin} -u train.py --iteration {iteration}"
        
        print(f"🏋️  [Remote] Starting Training Iteration {iteration}...")
        
        full_cmd = f"ssh {self.flags} {self.target} \"{remote_cmd}\""
        
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"   [Linux]: {output.strip()}")

        if process.poll() == 0:
            print("✅ Remote Training Finished.")
            return True
        else:
            print(f"❌ Remote Training Failed:\n{process.stderr.read()}")
            raise RuntimeError("Remote script crashed.")