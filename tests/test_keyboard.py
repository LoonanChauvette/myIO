from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import patch

from myio.keyboard import _KeyboardQueue, normalize_key_name, resolve_key


class KeyboardQueueTests(unittest.TestCase):
    def test_normalizes_names_and_rejects_unknown_keys(self) -> None:
        self.assertEqual(normalize_key_name("A"), "a")
        self.assertEqual(normalize_key_name("left-ctrl"), "left_control")
        self.assertEqual(normalize_key_name("Esc"), "escape")
        self.assertEqual(resolve_key("Space"), (0x20,))
        with self.assertRaises(ValueError):
            normalize_key_name("not a key")

    def test_builds_mask_and_converts_press_and_release(self) -> None:
        calls: list[tuple[object, ...]] = []
        raw_events = [
            ({"Type": 0, "Keycode": 32, "Pressed": 1, "Time": 10.25}, 1),
            ({"Type": 0, "Keycode": 32, "Pressed": 0, "Time": 10.5}, 0),
        ]

        def psych_hid(*args: object) -> object:
            calls.append(args)
            if args[0] == "KbQueueFlush":
                return 2
            if args[0] == "KbQueueGetEvent":
                return raw_events.pop(0)
            return None

        with patch("myio.keyboard.PsychHID", psych_hid):
            keyboard = _KeyboardQueue({"space"})
            keyboard.start()
            events = keyboard.poll()
            keyboard.stop()

        create = calls[0]
        mask = cast(list[int], create[2])
        self.assertIsInstance(mask, list)
        self.assertEqual(mask[31], 1)
        self.assertEqual(sum(mask), 1)
        self.assertEqual(
            [
                (event.key, event.pressed, event.timestamp, event.raw_code)
                for event in events
            ],
            [("space", True, 10.25, 32), ("space", False, 10.5, 32)],
        )
        self.assertEqual(
            [call[0] for call in calls],
            [
                "KbQueueCreate",
                "KbQueueStart",
                "KbQueueFlush",
                "KbQueueGetEvent",
                "KbQueueGetEvent",
                "KbQueueStop",
                "KbQueueRelease",
            ],
        )


if __name__ == "__main__":
    unittest.main()
