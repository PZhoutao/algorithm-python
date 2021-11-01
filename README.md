# algorithm-python
Simple is better than complex.

## Built-in Functions
* abs(x)/max()/min() 绝对值/最大值/最小值 max(iterable) or max(arg1, *args), 还可以和sorted一样提供key
* all(iterable)/any(iterable)
* bin()/oct()/hex() 10进制转二进制(0b开头)/八进制(0o开头)/十六进制(0x开头)
* chr()/ord() 字符转ASCII或ASCII转字符
* divmod(a, b) return (a // b, a % b)
* enumerate(iterable, start=0)
* eval(expression) x = 1; eval('x+1') => 2
* filter(function, iterable)
* format(value[, format_spec]) format(x, "10.5f")
* frozenset([iterable]) return a frozenset
* globals()/locals() return a dictionary of global/local variables
* hash() used to quickly compare dictionary keys during a dictionary lookup
* id() 返回对象的唯一标识符
* int(x, base=10))/float()/str() 转整数(可自定义进制, int("10000", 2) => 16)/转浮点数(float('inf') float('-inf') 无限大 无限小)/转字符串
* isinstance(object, classinfo)/issubclass(class, classinfo)
* iter()/next() return an iterator/retrieve the next item
* len(iterable)
* map(function, iterable) reduce在functools里
* pow(base, exp[, mod]) more efficient than pow(base, exp) % mod
* print(*objects, sep=' ', end='\n')
* range(stop)/range(start, stop[, step])
* reversed(seq) return a reverse iterator
* round(number[,ndigits])
* sorted(iterable, key=None, reverse=False)
* sum(iterable, start=0)
* zip(iteralble, ...)

## The Python Standard Library
### Math
