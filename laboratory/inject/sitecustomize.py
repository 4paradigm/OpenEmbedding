# install the apport exception handler if available
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os
if os.environ.get('HYPEREMBEDDING_INJECT_TENSORFLOW', None) == '1':
    import sys
    sys.argv=[""]
    import openembedding_inject_tensorflow
