"""
LeetCode data structure reference for local learning and RAG search.

LeetCode defines these node classes in its judge environment, so many copied
solutions refer to ListNode, TreeNode, or Node without defining them.

Input formats used by LeetCode:

- Linked list:
  - [] means an empty linked list, so the head is None in Python.
  - [1, 2, 3] means 1 -> 2 -> 3.

- Binary tree:
  - [] means an empty binary tree, so the root is None in Python.
  - [1, null, 2, 3] is a level-order representation.
  - null marks a missing child.
  - In local Python code, use None instead of JSON-style null.

LeetCode also reuses the class name Node in different problems. A random-list
Node, graph Node, and N-ary-tree Node have different fields. This local
reference uses RandomListNode, GraphNode, and NaryNode to keep them separate.

This file is intentionally simple. It is useful both as executable reference
code and as searchable context for a small Coding RAG corpus.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


class ListNode:
    """Singly linked list node used by LeetCode linked-list problems."""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

    def __repr__(self) -> str:
        return f"ListNode({self.val})"


class TreeNode:
    """Binary tree node used by LeetCode binary-tree problems."""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.val})"


class RandomListNode:
    """Node used by 'Copy List with Random Pointer' style problems."""

    def __init__(
        self,
        val: int = 0,
        next: Optional["RandomListNode"] = None,
        random: Optional["RandomListNode"] = None,
    ):
        self.val = val
        self.next = next
        self.random = random


class GraphNode:
    """Node used by graph clone/traversal problems."""

    def __init__(self, val: int = 0, neighbors: Optional[list["GraphNode"]] = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class NaryNode:
    """Node used by N-ary tree problems."""

    def __init__(self, val: int = 0, children: Optional[list["NaryNode"]] = None):
        self.val = val
        self.children = children if children is not None else []


def build_linked_list(values: list[int]) -> Optional[ListNode]:
    """Convert LeetCode linked-list format [1, 2, 3] into ListNode objects."""
    dummy = ListNode()
    current = dummy

    for value in values:
        current.next = ListNode(value)
        current = current.next

    return dummy.next


def linked_list_to_list(head: Optional[ListNode]) -> list[int]:
    """Convert a ListNode chain back into LeetCode linked-list format."""
    values: list[int] = []
    current = head

    while current is not None:
        values.append(current.val)
        current = current.next

    return values


def build_binary_tree(values: list[Any]) -> Optional[TreeNode]:
    """Convert LeetCode level-order tree format into TreeNode objects.

    Example:
        [1, None, 2, 3] becomes a tree whose root is 1, right child is 2,
        and node 2 has left child 3.
    """
    if not values:
        return None

    root_value = values[0]
    if root_value is None:
        return None

    root = TreeNode(root_value)
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
    """Convert a TreeNode tree back into LeetCode level-order format."""
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


def example_usage() -> None:
    linked = build_linked_list([1, 2, 3])
    assert linked_list_to_list(linked) == [1, 2, 3]

    tree = build_binary_tree([1, None, 2, 3])
    assert binary_tree_to_list(tree) == [1, None, 2, 3]
