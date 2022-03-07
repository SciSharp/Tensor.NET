## Organization of Layout in c++ part

### The effect of ```Shape```

In the c++ part of **Num.NET**, struct ```Shape``` is defined in ```core/base/include/layout.h```. It is an important and essential class which indicates the organization of the data in ```NDArray ```.

In ```NDArray```, all data are saved as a pointer with a buffer, which could be considered as an one-dimensional array. It's the ```Shape``` in ```NDArray``` that decides the organization of data. For instance, the data in ```NDArray``` are ```[1, 2, 3, 4, 5, 6]```. If the shape is ```[2, 3]```, it's a matrix with the data below:

```
[[1, 2, 3],
 [4, 5, 6]]
```

On the contrary, if the shape is ```[1, 3, 2]```, the matrix is in fact like this:

```
[[[1, 2],
  [3, 4],
  [5, 6]]]
```

---

### The Organization of Struct ```Shape```

In struct ```Shape```, there is a member called ```shape``` which indicates the shape.

```
size_t shape[MAX_NDIM];
```

Let's assume that ```MAX_NDIM = 4```, then what happens when we try to declare a ```Shape``` struct with initializer list ```{1, 3, 32, 64}```?

The shape information in the initializer list will be transferred to member ```shape``` in struct ```Shape``` as below.

| Index | 3 | 2 | 1 | 0 |
| --- | --| --| --| --|
| Value | 1 | 3 | 32 | 64 |

Note that the example below is based on the assumption that ```Shape::MAX_NDIM``` = 4.

---

### The structure of ```Layout```

The ```Layout``` class is a derived class of ```Shape```. It contains type and format information, as listed below.

```
DType dtype;
Format format;
```

More importantly, it contains a member named ```stride```, which means "how many elements will be passed when the value of this dim add one".

For easier understanding, let's see an example.

There is an ```NDArray``` with shape ```{2, 3, 4, 5}```, then its default stride is ```{60, 20, 5, 1}```.

The ```stride``` is useful in calculating the index of original array in ```NDArray```, which avoids duplicated multiply operation. Furthermore, when broadcasting, modifying stride instead of modifying data provides better performance.