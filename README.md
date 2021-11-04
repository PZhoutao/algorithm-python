# algorithm-python
Simple is better than complex.

## The Python Standard Library

### Built-in Functions
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

### math
Constants

* math.pi
* math.e
* math.inf/-math.inf equivalent to float('inf')
* math.nan A floating-point “not a number” (NaN) value. Equivalent to the output of float('nan')

Funcitons

* math.ceil(x)/floor(x)/trunc(x) 取整 trunc = floor
* math.degrees(x)/math.radians(x)
* math.exp(x) e^x
* math.gcd(a, b) 与b同号，任一为0时返回0 least common multiple = a*b/gcd(a,b)
* math.log(x)/math.log2(x)/math.log10(x)
* math.modf(x) 返回浮点数x的小数和整数部分 modf(32.5)=(0.5,32)
* math.pow(x,y)
* math.sin(x)/math.asin(x)/math.cos(x)/math.acos(x)/math.tan(x)/math.atan(x)
* math.sqrt(x)
* math.isnan/math.isinf/math.isfinite isfinite当检测数为inf和nan时返回False

### statistics
* statistics.mean() 算术平均数
* statistics.geometric_mean() 几何平均数
* statistics.harmonic_mean() 调和均值
* statistics.median() 中位数 [1, 3, 5, 7] => 4.0
* statistics.median_low() 低中位数 [1, 3, 5, 7] => 3
* statistics.median_high() 高中位数 [1, 3, 5, 7] => 5
* statistics.mode() 众数
* statistics.pstdev(data, mu=None) 总体标准差 
* statistics.pvariance(data, mu=None) 总体方差，除以n
* statistics.stdev(data, xbar=None) 样本标准差
* statistics.variance(data, xbar=None) 样本方差，除以n-1

### random
* random.seed(s)
* random.randrange(stop)/random.randrange(start, stop[, step]) randomly select **one element** from range, equivalent to choice(range(start, stop, step)), but doesn’t build a range object.
* random.randint(a, b) return a random integer N such that a <= N <= b
* random.choice(seq) return a random element from seq
* random.choices(population, weights=None, cum_weights=None, k=1) return k-sized list of element from the population with replacement
* random.shuffle(x) shuffle the sequence x in place
* random.sample(population, k, counts=None) sample without replacement
* random.random() [0.0, 1.0)
* random.uniform(a, b) a random floating point number N such that a <= N <= b
* random.gauss(mu, sigma)

### re
* non-greedy qualifier *?, +?, ??
* re.compile(pattern) p = re.compile(pattern); result = p.match(string) 等于 result = re.match(pattern string) reuse下节省时间
* re.I/re.IGNORECASE flag, perform case-insensitive matching
* re.search(pattern, string, flags=0)
* re.match(pattern, string, flags=0) from beginning of string
* re.split(pattern, string, maxsplit=0, flags=0)
* re.findall(pattern, string, flags=0)/re.finditer(pattern, string, flags=0) return a list/iterator of non-overlapping matches of pattern in string
* re.sub(pattern, repl, string, count=0, flags=0)

### string
Constants

* string.ascii_letters/string.ascii_lowercase/string.ascii_uppercase/string.digits
* string.punctuation/string.whitespace

Methods

* str.capitalize()/str.lower()/str.upper()/str.islower()/str.isupper()
* str.count(sub[,start[,end]])
* str.find(sub[,start[,end]])
* str.format()
* str.index(sub[,start[,end]])
* str.join(iterable)
* str.replace(old, new, [count])
* str.split(sep=None, maxsplit=-1)
* str.strip([chars])
* str.title()
* str.translate(table)

### copy
* copy.copy(x) shallow copy
* copy.deepcopy(x)

### datetime
* date(year, month, day): 日期模型
* time(hour, minute, second, microsecond, tzinfo): 时间模型
* datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
* datetime.now()
* timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)

### time
* time.sleep(secs)
* time.time() seconds since the epoch

## Functional Programming
### functools
The functools module is for higher-order functions: functions that act on or return other functions.

* @cache Creating a thin wrapper around a dictionary lookup for the function arguments. Because it never needs to evict old values, this is smaller and faster than lru_cache() with a size limit.
* @lru_cache(maxsize=128, typed=False)
* functools.reduce(function, iterable[, initializer])
* partial()

### itertools
无穷迭代器

* itertools.count(start=0, step=1)
* itertools.cycle(iterable)
* itertools.repeat(object[, times]) map(pow, range(10), repeat(2)) 提供常量
有限迭代器
* itertools.accumulate(iterable[, func, initial=None]) Make an iterator that returns accumulated sums, or accumulated results of other binary functions. accumulate([1,2,3,4,5], operator.mul) 
* itertools.chain(*iterables) Chain iterables. chain('ABC', 'DEF') --> A B C D E F
* itertools.compress(data, selectors) compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
* itertools.dropwhile/takewhile(predicate, iterable) Drops/take elements from the iterable as long as the predicate is true
* itertools.filterfalse(predicate, iterable) 与filter相反
* itertools.groupby(iterable, key=None) 需要提前sort iterable by key, 才能把所有相同key的都放到一起
* itertools.islice(iterable, stop)/(iterable, start, stop[, step]) 返回iterable里的元素
* itertools.starmap(function, iterable) Used instead of map() when argument parameters are already grouped in tuples("pre-zipped"). starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
* itertools.tee(iterable, n=2) Return n independent iterators from a single iterable.
* itertools.zip_longest(*iterables, fillvalue=None)

排列组合迭代器
* itertools.product(*iterables, repeat=1）product('ABC', 'xy') --> Ax Ay Bx By Cx Cy product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
* itertools.permutations(iterable, r=None) permutations('ABC', 2) --> AB AC BA BC CA CB
* itertools.combinations(iterable, r) combinations('ABC', 2) --> AB AC BC
* itertools.combinations_with_replacement(iterable, r) combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC

## Data Structure and Algorithms
### bisect
* bisect.bisect_left(a, x, lo=0, hi=len(a), *, key=None) lo and hi may be used to specify a subset of the list, by default the entire list is used
* bisect.bisect_right(...)/bisect.bisect(...) 感觉是一样的
* bisect.insort_left(a, x, lo=0, hi=len(a), *, key=None) insert x in a in sorted order
* bisect.insort_right(...)/bisect.insort(...)

### collections.Counter
A dict subclass for counting hashable objects.

```
c = Counter()
c = Counter('gallahad')
c = Counter({'red': 4, 'blue': 2})
c = Counter(cats=4, dogs=8)

c['sausage'] = 0
del c['sausage'] 
```
* counter.elements() Return an iterator over elements repeating each as many times as its count.
* counter.most_common([n])
* counter.subtract([iterable-or-mapping])
* counter.total()
* counter.update([iterable-or-mapping])) Like dict.update() but adds counts instead of replacing them.

### collections.defaultdict
```
d = defaultdict(list)
d[k].append(v)

```

### collections.deque
Deques support thread-safe, memory efficient appends and pops from either side of the deque with approximately the same O(1) performance in either direction. Implemented by a bi-directional linked list.

* collections.deque([iterable[, maxlen]]) Returns a new deque object initialized left-to-right.
* d.append(x)/d.appendleft(x) Add x to the right/left side of the deque.
* d.clear()
* d.copy()
* d.extend(iterable)/d.extendleft(iterable)
* d.pop()/d.popleft() Remove and return an element from the right/left side of the deque.
* d.rotate(n=1) Rotate the deque n steps to the right. If n is negative, rotate to the left.

```
if d:
    # not empty
else:
    # empty
```

### collections.namedtuple
* collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)

```
Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)
x, y = p
p.x + p.y 
```

### heapq
An implementation of the priority queue algorithm by heap.
Heaps are binary trees for which every parent node has a value less than or equal to any of its children (min heap).

* heapq.heappush(heap, item)/heapq.heappop(heap)
* heapq.heapify(x)
* heapq.nlargest(n, iterable, key=None)/heapq.nsmallest(n, iterable, key=None)