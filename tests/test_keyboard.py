from __future__ import annotations

import time
import unittest
from typing import cast
from unittest.mock import patch

from myio.keyboard import KeyboardQueue, normalize_key_name, resolve_key


class KeyboardQueueTests(unittest.TestCase):
    def test_normalizes_names_and_rejects_unknown_keys(self) -> None:
        self.assertEqual(normalize_key_name("A"), "a")
        self.assertEqual(normalize_key_name("left-ctrl"), "left_control")
        self.assertEqual(normalize_key_name("Esc"), "escape")
        self.assertEqual(resolve_key("Space"), (0x20,))
        with self.assertRaises(ValueError):
            normalize_key_name("not a key")

    def test_builds_mask_and_lifecycle_calls(self) -> None:
        calls: list[tuple[object, ...]] = []

        def psych_hid(*args: object) -> object:
            calls.append(args)
            if args[0] == "KbQueueGetEvent":
                return None, 0
            return None

        with (
            patch("myio.keyboard.PsychHID", psych_hid),
            patch("myio.keyboard.GetSecs", return_value=1.0),
        ):
            keyboard = KeyboardQueue({"space"})
            keyboard.start()
            self.assertTrue(keyboard.is_started())
            keyboard.stop()
            self.assertTrue(keyboard.is_stopped())
            self.assertFalse(keyboard.is_started())

        create = calls[0]
        self.assertEqual(create[0], "KbQueueCreate")
        mask = cast(list[int], create[2])
        self.assertEqual(mask[31], 1)
        self.assertEqual(sum(mask), 1)

        names = [call[0] for call in calls]
        self.assertEqual(names[0], "KbQueueCreate")
        self.assertEqual(names[1], "KbQueueStart")
        self.assertEqual(names[-2], "KbQueueStop")
        self.assertEqual(names[-1], "KbQueueRelease")
        self.assertTrue(all(name == "KbQueueGetEvent" for name in names[2:-2]))

    def test_converts_press_and_release(self) -> None:
        keyboard = KeyboardQueue({"space"})
        press = keyboard._to_key_event(
            {
                "Type": 0,
                "Keycode": 32,
                "Pressed": 1,
                "Time": 10.25,
                "CookedKey": 0,
                "ButtonStates": 0,
                "Motion": 0,
                "X": 0,
                "Y": 0,
                "NormX": 0,
                "NormY": 0,
                "Valuators": cast(object, None),
            }
        )
        release = keyboard._to_key_event(
            {
                "Type": 0,
                "Keycode": 32,
                "Pressed": 0,
                "Time": 10.5,
                "CookedKey": 0,
                "ButtonStates": 0,
                "Motion": 0,
                "X": 0,
                "Y": 0,
                "NormX": 0,
                "NormY": 0,
                "Valuators": cast(object, None),
            }
        )
        assert press is not None
        assert release is not None
        self.assertEqual(
            (press.key, press.pressed, press.timestamp, press.raw_code),
            ("space", True, 10.25, 32),
        )
        self.assertEqual(
            (release.key, release.pressed, release.timestamp, release.raw_code),
            ("space", False, 10.5, 32),
        )

    def test_pending_events_drains_queue(self) -> None:
        keyboard = KeyboardQueue({"space"})
        event = keyboard._to_key_event(
            {
                "Type": 0,
                "Keycode": 32,
                "Pressed": 1,
                "Time": 1.0,
                "CookedKey": 0,
                "ButtonStates": 0,
                "Motion": 0,
                "X": 0,
                "Y": 0,
                "NormX": 0,
                "NormY": 0,
                "Valuators": cast(object, None),
            }
        )
        assert event is not None
        keyboard._queue.put(event)

        drained = list(keyboard.pending_events)
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].key, "space")
        self.assertEqual(list(keyboard.pending_events), [])

    def test_worker_failure_sets_error(self) -> None:
        def psych_hid(*args: object) -> object:
            if args[0] == "KbQueueGetEvent":
                raise RuntimeError("hid boom")
            return None

        with (
            patch("myio.keyboard.PsychHID", psych_hid),
            patch("myio.keyboard.GetSecs", return_value=1.0),
        ):
            keyboard = KeyboardQueue({"space"})
            keyboard.start()
            deadline = time.monotonic() + 1.0
            while keyboard.error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsInstance(keyboard.error, RuntimeError)
            self.assertIn("hid boom", str(keyboard.error))
            keyboard.stop()

    def test_cannot_restart_after_stop(self) -> None:
        def psych_hid(*args: object) -> object:
            if args[0] == "KbQueueGetEvent":
                return None, 0
            return None

        with (
            patch("myio.keyboard.PsychHID", psych_hid),
            patch("myio.keyboard.GetSecs", return_value=1.0),
        ):
            keyboard = KeyboardQueue({"space"})
            keyboard.start()
            keyboard.stop()
            with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                keyboard.start()


if __name__ == "__main__":
    unittest.main()
