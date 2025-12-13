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

"""Utility functions for accessing common memory values"""
from typing import List, Optional

from pydantic import BaseModel

from ..models.common import Memory, MemoryScope
from .memory import get_memory_value, set_memory_value


class UserIdentity(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    preferredName: Optional[str] = None
    id: Optional[str] = None


class UserLocation(BaseModel):
    city: Optional[str] = None
    country: Optional[str] = None


def set_user_identity(
    memories: List[Memory],
    *,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    preferred_name: Optional[str] = None,
    id: Optional[str] = None,
) -> None:
    has_identity, identity = get_memory_value(
        memories, "userIdentity", scope=MemoryScope.PUBLIC
    )
    if has_identity:
        try:
            identity = UserIdentity(**identity)
        except Exception:
            identity = UserIdentity()

        if first_name:
            identity.firstName = first_name
        if last_name:
            identity.lastName = last_name
        if preferred_name:
            identity.preferredName = preferred_name
        if id:
            identity.id = id
    else:
        identity = UserIdentity(
            firstName=first_name,
            lastName=last_name,
            preferredName=preferred_name,
            id=id,
        )

    set_memory_value(
        memories, "userIdentity", identity.dict(), scope=MemoryScope.PUBLIC
    )


def get_user_identity(
    memories: List[Memory],
) -> Optional[UserIdentity]:
    has_identity, identity = get_memory_value(
        memories, "userIdentity", scope=MemoryScope.PUBLIC
    )
    if has_identity:
        try:
            return UserIdentity(**identity)
        except Exception:
            return None

    return None


def set_user_location(
    memories: List[Memory],
    *,
    city: Optional[str] = None,
    country: Optional[str] = None,
) -> None:
    has_location, location = get_memory_value(
        memories, "userLocation", scope=MemoryScope.PUBLIC
    )
    if has_location:
        try:
            location = UserLocation(**location)
        except Exception:
            location = UserLocation()

        if city:
            location.city = city
        if country:
            location.country = country
    else:
        location = UserLocation(
            city=city,
            country=country,
        )

    set_memory_value(
        memories, "userLocation", location.dict(), scope=MemoryScope.PUBLIC
    )


def get_user_location(
    memories: List[Memory],
) -> Optional[UserLocation]:
    has_location, location = get_memory_value(
        memories, "userLocation", scope=MemoryScope.PUBLIC
    )
    if has_location:
        try:
            return UserLocation(**location)
        except Exception:
            return None

    return None
