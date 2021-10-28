# Start

## Parameter Server in Process
```python
import openembedding.tensorflow as embed
```

## Remote Parameter Server

### Master
```python
import time
import openembedding as embed
master = embed.Master()
time.sleep(10) # Wait
```

### Parameter Server
```python
import openembedding as embed
embed.flags.master_endpoint = '{ip}:{port}'
_server = embed.Server()
_server.join()
```

### Worker
```python
import openembedding.tensorflow as embed
embed.flags.master_endpoint = '{ip}:{port}'
embed.flags.wait_num_servers = num_servers
```
