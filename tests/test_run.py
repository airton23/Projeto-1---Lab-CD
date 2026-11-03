import unittest
from unittest.mock import Mock, patch

from config import DATA_PATH
from run import main


class EntryPointTests(unittest.TestCase):
    def test_default_execution_uses_waitress(self):
        application = Mock()
        with patch("app.create_app", return_value=application) as create, \
             patch("waitress.serve") as serve, \
             patch.dict("os.environ", {"HOST": "127.0.0.1", "PORT": "5050"}), \
             patch("builtins.print"):
            main([])
        create.assert_called_once_with(None)
        serve.assert_called_once_with(application, host="127.0.0.1", port=5050, threads=4)
        application.run.assert_not_called()

    def test_development_enables_reload_and_disables_static_cache(self):
        application = Mock()
        with patch("app.create_app", return_value=application) as create, \
             patch("waitress.serve") as serve, \
             patch.dict("os.environ", {"HOST": "127.0.0.1", "PORT": "5050"}), \
             patch("builtins.print"):
            main(["--dev"])
        self.assertEqual(create.call_args.args[0], {
            "DEBUG": True, "TEMPLATES_AUTO_RELOAD": True, "SEND_FILE_MAX_AGE_DEFAULT": 0,
        })
        application.run.assert_called_once_with(
            host="127.0.0.1", port=5050, debug=True, use_debugger=False,
            extra_files=[str(DATA_PATH)],
        )
        serve.assert_not_called()
