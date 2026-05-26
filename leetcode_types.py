"""
本地 LeetCode 解法运行兼容层

LeetCode 在其在线评测环境中注入了 ListNode、TreeNode 和 Node 等类。
许多复制的解决方案文件因此省略了这些定义。如果你想在 LeetCode 之外
运行本地解决方案，可以从解决方案文件中导入此模块。

通用的 Node 类设计得比较灵活，因为 LeetCode 在不同问题家族中重用
同一名称：随机链表、图、N 叉树、带 next 指针的二叉树以及多级双向链表。
"""

from __future__ import annotations

import bisect
import collections
import heapq
import itertools
import math
from collections import Counter, defaultdict, deque
from bisect import bisect_left, bisect_right
from functools import cache, lru_cache
from heapq import heapify, heappop, heappush
from math import inf
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


class ListNode:
    """LeetCode 链表节点类"""
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        """初始化链表节点
        
        Args:
            val: 节点值
            next: 下一个节点的引用
        """
        self.val = val
        self.next = next

    def __repr__(self) -> str:
        return f"ListNode({self.val})"


class TreeNode:
    """LeetCode 二叉树节点类"""
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        """初始化二叉树节点
        
        Args:
            val: 节点值
            left: 左子节点引用
            right: 右子节点引用
        """
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.val})"


class Node:
    """
    LeetCode 通用节点类
    
    用于多种数据结构：随机链表、图、N 叉树、带 next 指针的二叉树、多级双向链表等
    """
    def __init__(
        self,
        val: int = 0,
        *args: Any,
        next: Optional["Node"] = None,
        random: Optional["Node"] = None,
        neighbors: Optional[list["Node"]] = None,
        children: Optional[list["Node"]] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        prev: Optional["Node"] = None,
        child: Optional["Node"] = None,
    ):
        """初始化通用节点
        
        Args:
            val: 节点值
            *args: 可变参数，用于不同的数据结构类型
            next: 下一个节点引用（链表、带 next 指针的二叉树等）
            random: 随机指针引用（随机链表）
            neighbors: 邻居节点列表（图）
            children: 子节点列表（N 叉树）
            left: 左子节点引用（二叉树）
            right: 右子节点引用（二叉树）
            prev: 前一个节点引用（双向链表）
            child: 子节点引用（多级链表）
        """
        self.val = val

        # 根据参数类型和数量推断节点类型
        if len(args) == 1 and isinstance(args[0], list):
            neighbors = args[0]
            children = args[0]
        elif len(args) == 1:
            next = args[0]
        elif len(args) == 2:
            next, random = args
            left, right = args
        elif len(args) == 3:
            left, right, next = args
            prev = args[0]
            child = args[2]
        elif len(args) >= 4:
            prev, next, child, random = args[:4]

        self.next = next
        self.random = random
        self.neighbors = neighbors if neighbors is not None else []
        self.children = children if children is not None else []
        self.left = left
        self.right = right
        self.prev = prev
        self.child = child

    def __repr__(self) -> str:
        return f"Node({self.val})"


# 类型别名，用于不同的 LeetCode 问题类型
RandomListNode = Node  # 随机链表节点
GraphNode = Node  # 图节点
NaryNode = Node  # N 叉树节点


def build_linked_list(values: list[int]) -> Optional[ListNode]:
    dummy = ListNode()
    current = dummy

    for value in values:
        current.next = ListNode(value)
        current = current.next

    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> list[int]:
    values: list[int] = []
    current = head

    while current is not None:
        values.append(current.val)
        current = current.next

    return values


def build_binary_tree(values: list[Any]) -> Optional[TreeNode]:
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    queue: deque[TreeNode] = deque([root])
    index = 1

    while queue and index < len(values):
        node = queue.popleft()

        if index < len(values) and values[index] is not None:
            node.left = TreeNode(values[index])
            queue.append(node.left)
        index += 1

        if index < len(values) and values[index] is not None:
            node.right = TreeNode(values[index])
            queue.append(node.right)
        index += 1

    return root


def binary_tree_to_list(root: Optional[TreeNode]) -> list[Any]:
    if root is None:
        return []

    values: list[Any] = []
    queue: deque[Optional[TreeNode]] = deque([root])

    while queue:
        node = queue.popleft()

        if node is None:
            values.append(None)
            continue

        values.append(node.val)
        queue.append(node.left)
        queue.append(node.right)

    while values and values[-1] is None:
        values.pop()

    return values


__all__ = [
    "Any",
    "bisect",
    "bisect_left",
    "bisect_right",
    "collections",
    "Counter",
    "Deque",
    "Dict",
    "GraphNode",
    "List",
    "ListNode",
    "NaryNode",
    "Node",
    "Optional",
    "RandomListNode",
    "Set",
    "TreeNode",
    "Tuple",
    "binary_tree_to_list",
    "build_binary_tree",
    "build_linked_list",
    "cache",
    "defaultdict",
    "deque",
    "heapq",
    "heapify",
    "heappop",
    "heappush",
    "inf",
    "itertools",
    "linked_list_to_list",
    "lru_cache",
    "math",
]
