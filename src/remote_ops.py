# src/remote_ops.py
import subprocess
import os
import sys

class RemoteManager:
    def __init__(self, hostname, username, ssh_key_path, remote_project_root):
        """
        A pure utility class for handling SSH/SCP operations.
        It knows 'how' to connect, but doesn't care 'what' you are running.
        """
        self.target = f"{username}@{hostname}"
        self.key_path = os.path.expanduser(ssh_key_path)
        self.remote_root = remote_project_root
        
        # Standard SSH flags for non-interactive automation
        # -4: Force IPv4
        # -o BatchMode=yes: Don't ask for passwords (fail if key missing)
        # -o StrictHostKeyChecking=no: Don't block on new fingerprints
        self.flags = f"-4 -i {self.key_path} -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    def run_command(self, command):
        """
        Executes a command on the Linux machine and waits for it to finish.
        Best for quick checks (e.g., "does file exist?").
        
        Returns: (exit_code, stdout_string, stderr_string)
        """
        # Wrap command in quotes to pass it as a single argument to SSH
        full_cmd = f"ssh {self.flags} {self.target} \"{command}\""
        
        result = subprocess.run(
            full_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def stream_command(self, command, env_vars=None):
        """
        Executes a command and streams the output line-by-line to the Mac console.
        Best for long-running jobs (e.g., Training) so the terminal doesn't freeze.
        
        Args:
            command: The command string to run on Linux.
            env_vars: Optional dict of environment variables to inject (e.g., {'TAG': 'Remote'}).
        """
        # 1. Prepend Environment Variables if provided
        prefix = ""
        if env_vars:
            # Result: "export VAR1='Val1'; export VAR2='Val2'; "
            prefix = " ".join([f"export {k}='{v}';" for k, v in env_vars.items()])
        
        # 2. Construct the full SSH command
        # We cd to remote_root first to ensure relative paths work
        remote_cmd_str = f"{prefix} cd {self.remote_root} && {command}"
        full_cmd = f"ssh {self.flags} {self.target} \"{remote_cmd_str}\""
        
        print(f"📡 [Stream] Executing on {self.target}...")

        # 3. Open the process
        process = subprocess.Popen(
            full_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 4. Stream stdout line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print exactly what Linux sends, stripping double newlines
                print(f"   [Linux]: {output.rstrip()}")

        # 5. Check final status
        if process.poll() == 0:
            return True
        else:
            # Capture any remaining error text
            err = process.stderr.read()
            print(f"❌ Remote Command Failed: {err}")
            return False

    def sync_file(self, local_path, relative_remote_path):
        """
        Uploads a local file to the Linux machine (Mac -> Linux).
        """
        # Clean paths
        remote_full_path = f"{self.remote_root}/{relative_remote_path}".replace("//", "/")
        remote_dir = os.path.dirname(remote_full_path)

        # 1. Ensure the destination folder exists
        self.run_command(f"mkdir -p {remote_dir}")

        # 2. SCP Upload
        print(f"🚀 Uploading: {os.path.basename(local_path)}")
        cmd = f"scp {self.flags} \"{local_path}\" {self.target}:{remote_full_path}"
        
        try:
            subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"❌ Upload failed: {e}")
            raise

    def retrieve_file(self, relative_remote_path, local_destination):
        """
        Downloads a file from the Linux machine (Linux -> Mac).
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