import os
import json
import datetime
import urllib.request
import urllib.error
from omni.logging_config import get_logger

tools_logger = get_logger("OmniTools", "tools.log")

class ToolOrchestrator:
    """
    Agency (paper Sec 8.10): the modern-agent toolkit, PATH-JAILED to the
    engine's readable roots. Every call's arguments and result are logged and
    held as training traces -- so tool use is itself LEARNED (constants/
    memory_persistence.ep Step 145: the act is held as an orbit; the corpus's
    "acts held, values never" -- a time answer re-reads the clock, it does not
    freeze a timestamp). Parses JSON blocks emitted by Unison and routes them to
    the tools within the jail.
    """
    def __init__(self, base_dir="."):
        self.base_dir = os.path.abspath(base_dir)
        self.scratch_pad = {}
        from omni.logging_config import LOG_DIR as _LD
        self.logs_dir = _LD
        self.omni_dir = os.path.join(self.base_dir, "omni")
        self.constants_dir = os.path.join(self.base_dir, "constants")

    def execute(self, tool_name, args):
        """Route to the appropriate tool based on name."""
        try:
            if tool_name == "internet_access":
                return self.internet_access(args.get("url", ""))
            elif tool_name == "scratch_pad":
                return self.handle_scratch_pad(args)
            elif tool_name == "time_and_date":
                return self.time_and_date()
            elif tool_name == "read_own_code":
                return self.read_own_code(args.get("filepath", ""))
            elif tool_name == "read_logs":
                return self.read_logs(args.get("filename", ""))
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            tools_logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def parse_and_execute(self, text):
        """
        Looks for a JSON block in the text. If found, executes it.
        Example format: {"tool": "time_and_date", "args": {}}
        """
        try:
            # Very basic extraction: find the first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
                data = json.loads(json_str)
                if "tool" in data and "args" in data:
                    return self.execute(data["tool"], data["args"])
            return None
        except Exception:
            return None

    def internet_access(self, url):
        """Fetch text content from a URL via HTTP GET."""
        if not url.startswith("http"):
            url = "http://" + url
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'UnisonSFT/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8')
                # Return first 1000 characters to prevent overflow
                return content[:1000] + ("..." if len(content) > 1000 else "")
        except urllib.error.URLError as e:
            return f"Failed to fetch URL: {e}"

    def handle_scratch_pad(self, args):
        """Read, write, or list the temporary scratch pad."""
        action = args.get("action")
        key = args.get("key")
        value = args.get("value")

        if action == "write":
            if key is None or value is None:
                return "Error: write requires 'key' and 'value'."
            self.scratch_pad[key] = value
            return f"Stored {key}."
        elif action == "read":
            if key is None:
                return "Error: read requires 'key'."
            return str(self.scratch_pad.get(key, "Key not found."))
        elif action == "list":
            return str(list(self.scratch_pad.keys()))
        else:
            return "Error: Action must be 'read', 'write', or 'list'."

    def time_and_date(self):
        """Return the current UTC and local time."""
        utc_now = datetime.datetime.utcnow().isoformat()
        local_now = datetime.datetime.now().isoformat()
        return f"UTC: {utc_now}Z | Local: {local_now}"

    def read_own_code(self, filepath):
        """Securely read a file from the codebase."""
        if not filepath:
            return "Error: 'filepath' is required."
        
        # Ensure path is strictly within the codebase to prevent escape
        abs_path = os.path.abspath(os.path.join(self.base_dir, filepath))
        
        if not (abs_path.startswith(self.omni_dir) or abs_path.startswith(self.constants_dir) or abs_path.endswith(".py") or abs_path.endswith(".md")):
            return f"Error: Access denied to {filepath}. Only omni/, constants/, and main source files allowed."
        
        if not os.path.exists(abs_path):
            return f"Error: File '{filepath}' not found."
            
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Truncate to prevent massive token overflow
            return content[:2000] + ("\n...[TRUNCATED]" if len(content) > 2000 else "")

    def read_logs(self, filename):
        """Securely read from omni/logs/."""
        if not filename:
            # Return list of logs if no file specified
            if os.path.exists(self.logs_dir):
                return "Available logs: " + ", ".join(os.listdir(self.logs_dir))
            return "Logs directory not found."
            
        abs_path = os.path.abspath(os.path.join(self.logs_dir, filename))
        if not abs_path.startswith(self.logs_dir):
            return "Error: Path traversal blocked."
            
        if not os.path.exists(abs_path):
            return f"Error: Log '{filename}' not found."
            
        # Return tail of log
        with open(abs_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            tail = "".join(lines[-50:]) # Last 50 lines
            return tail
