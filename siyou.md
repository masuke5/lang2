# lang2

## データ型

### プリミティブ型

#### int
64bitの整数型

#### float
64bitの浮動小数点型

#### char
UTF-32でエンコードされた文字。

#### string
UTF-8でエンコードされた文字列。

#### ポインタ
型の前にアスタリスクを付ける。nullを格納することはできない。
```lang2
*int
*string
```

ポインタの指す値を変更するには "mut" を付ける。

```lang2
*mut int
*mut string
```

#### タプル
異なる型の値の集合。括弧で定義する。

```lang2
(int, float)
(string)
(5, 3.5)
```

### ユーザー定義型
datatypeキーワードを使用して定義する。

#### 代数的データ型
```lang2
datatype Option<T> = Some of T | None;
```

#### 構造体
```lang2
datatype Person = {
    name: string,
    age: int,
};
```

## リテラル
|             | 型     | 説明
|:------------|:-------|:------------------------
| 123         | int    | 10進数
| 123_456_789 | int    | アンダースコアを挿入できる
| 123.456     | float  | 小数
| 'a', 'あ'   | char   | 文字
| "Hello"     | string | 文字列

## 変数
letキーワードを使って定義する。

```lang2
let a: int = 5;
```

右辺から型が推測できる場合は型を省略できる。

```lang2
let a = 5;
```

変数はデフォルトで不変。

```lang2
let a = 5;
a := 7; # エラー
```

可変な変数はmutキーワードで定義する。

```lang2
let mut a = 5;
```

値を変更するには ":=" を使う。

```lang2
let mut a = 5;
a := 7;
```

変数は数値型の場合は "+=", "-=", "*=", "/=" を使うことができる。

```lang2
let mut a = 5;
a += 5;
a -= 5;
a *= 3;
```

letキーワードの左辺でパターンマッチをすることができる。

```lang2
let (x, y) = (3, 7);
```

## ブロック
";" で区切った文を波括弧で囲む。

```lang2
{
    let a = 5;
    add(5, a);
}
```

ブロックの中で宣言した変数は外から使用できない。

```lang2
{
    let a = 5;
}
add(5, a); # エラー
```

ブロックは式。ブロックの最後の式が結果。セミコロンは付けない。

```lang2
let a = {
    let a = 5;
    add(a, 7)
};
```

## 関数

### 定義
2つの数字を足す関数。
```lang2
fn add(a: int, b: int): int {
    a + b
}
```

副作用がある関数の場合は関数名の末尾に "!" を付ける。付いていない場合はエラーを出す。

```lang2
fn print_usage!() {
    println!("usage: lang2 -o [output] [input]");
}
```

戻り値の型を省略した場合は空のタプルを返す。次の関数は同じ意味。

```lang2
fn a(): () {
}

fn a() {
}
```

#### 早期リターン
return文を使う。

```lang2
fn abs(n: int) {
    if n < 0 {
        return -n;
    }

    n
}
```

#### 副作用がある関数
副作用がある関数は以下の条件を満たしたもの。

- 副作用がある関数を内部で使用している
- 可変な変数を変更している

### 呼び出し
コンマ区切りで引数を指定する。

```lang2
let a = add(3, 5);
print_usage!();
```

## ポインタ
"&" で変数から作成できる。

```lang2
let a = 5;
let b = &a;
```

"*" でポインタの指す値にアクセスできる。

```lang2
add(*b, 7);
```

ポインタの指す値を変更する場合は "mut" を付ける。

```lang2
let b = &mut a;
*b := 7;
```

nullを格納することはできない。

```lang2
let a: *int = null; # エラー
```

## 演算子と優先順位
| 優先順位 | 記号           | 説明
|:---------|:---------------|:---------------
| 1        | ::             | スコープ解決
| 1        | as             | キャスト
| 2        | a              | 変数
| 2        | a()            | 関数呼び出し
| 2        | a[]            | 添字
| 2        | a.b            | メンバアクセス, メソッド呼び出し
| 3        | &a             | アドレス取得
| 3        | *a             | 間接参照
| 4        | a*b            | 乗算
| 4        | a/b            | 除算
| 5        | a+b            | 加算
| 5        | a-b            | 減算
| 6        | a&lt;&lt;b     | 左シフト
| 6        | a&gt;&gt;b     | 右シフト
| 7        | a&b            | AND (ビット演算)
| 8        | a^b            | NOT (ビット演算)
| 9        | a&#x7C;b       | OR (ビット演算)
| 10       | a&lt;b         | 未満
| 10       | a&gt;b         | 超
| 10       | a&lt;=b        | 以下
| 10       | a&gt;=b        | 以上
| 11       | a==b           | 等しい
| 11       | a!=b           | 等しくない
| 12       | a&&b           | 論理積。遅延評価される
| 13       | a&#x7C;&#x7C;b | 論理和。遅延評価される

## 要加筆

## 例
```lang2
use std::math;

datatype Circle = {
    x: float,
    y: float,
    radius: float,
};

trait HasArea {
    fn area(*self): float;
}

impl HasArea for Circle {
    fn area(*self) -> float {
        math::PI * (self.radius * self.radius)
    }
}

fn get_area<T: HasArea>(a: T) -> float {
    a.area()
}

fn main() {
    print!("X: ");
    let x = input();
    print!("Y: ");
    let y = input();
    print!("Radius: ");
    let radius = input();

    let circle = Circle {
        x,
        y,
        radius,
    };

    println!("Area: {}", get_area(circle));
}
```
