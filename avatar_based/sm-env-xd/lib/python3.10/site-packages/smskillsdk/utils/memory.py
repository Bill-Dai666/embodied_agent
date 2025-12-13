# Copyright 2024 Soul Machines Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for working with memory arrays"""
from typing import Any, Dict, List, Tuple, Union

from smskillsdk.models.common import Memory, MemoryScope


def _has_valid_session_id_and_scope(
    memory: Memory,
    session_id: Union[str, None],
    scope: Union[MemoryScope, None],
    strict_session_id_match: bool = False,
) -> bool:
    if (
        strict_session_id_match or session_id is not None
    ) and memory.session_id != session_id:
        return False

    if (
        scope == MemoryScope.PUBLIC and memory.scope in [None, MemoryScope.PRIVATE]
    ) or (scope == MemoryScope.PRIVATE and memory.scope == MemoryScope.PUBLIC):
        return False

    return True


def serialize_memory(
    data: dict,
    session_id: Union[str, None] = None,
    scope: MemoryScope = MemoryScope.PRIVATE,
) -> List[Memory]:
    """Convert a Python dict into a list of Memory objects

    Key-value pairs will be converted into objects with these properties

    {
        "name": <key: Any>,
        "value": <value: Any>
        "session_id": <session_id: str> (optional)
        "scope": <PUBLIC | PRIVATE (default)>
    }
    """
    return [
        Memory(name=key, value=value, session_id=session_id, scope=scope)
        for key, value in data.items()
    ]


def deserialize_memory(
    memories: List[Memory],
    session_id: Union[str, None] = None,
    scope: Union[MemoryScope, None] = None,
) -> Dict[str, Any]:
    """Convert a list of memory objects to a dictionary

    It is assumed memory objects will have name/value attributes. If multiple memory
    objects share the same name values, the latest value in the list will be used. If
    a session_id is provided, this will be checked and used to filter out memories
    without a matching session_id.
    """
    return {
        memory.name: memory.value
        for memory in memories
        if _has_valid_session_id_and_scope(memory, session_id, scope)
    }


def set_memory_value(
    memories: List[Memory],
    key: str,
    value: Any,
    session_id: Union[str, None] = None,
    scope: MemoryScope = MemoryScope.PRIVATE,
) -> None:
    """Sets a value in the memory array corresponding to an optional session_id/scope

    Modifies the memories array in-place. A session ID can be provided to only modify
    a memory value with a matching session ID.
    """
    found = False

    for memory in memories:
        if not _has_valid_session_id_and_scope(memory, session_id, scope, True):
            continue

        if memory.name == key:
            memory.value = value
            found = True

    if not found:
        memories.append(
            Memory(name=key, value=value, session_id=session_id, scope=scope)
        )


def get_memory_value(
    memories: List[Memory],
    key: str,
    session_id: Union[str, None] = None,
    scope: Union[MemoryScope, None] = None,
) -> Tuple[bool, Any]:
    """Gets a value from a memory array corresponding to a session_id/scope"""
    for memory in memories:
        if not _has_valid_session_id_and_scope(memory, session_id, scope):
            continue

        if memory.name == key:
            return True, memory.value

    return False, None
