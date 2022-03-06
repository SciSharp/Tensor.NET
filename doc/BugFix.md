## BugFix Log

|  Description | Time | Solution |
| -------------- | --- | ----------- |
| The ```size_log``` of DTypes is wrong. It should be the log value of data length. This could cause allocating too much memory when initializing an NDArray. For instance, $2^{32}$ bits for ```NDArray<int>``` of size [1] | 2022/02/28 | Revise ```log_size``` to correct values. |