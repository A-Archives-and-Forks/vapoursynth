import unittest

import vapoursynth as vs


class PropDictTest(unittest.TestCase):
    def setUp(self):
        self.core = vs.core
        self.core.num_threads = 1
        self.frame = self.core.std.BlankClip().get_frame(0)
        self.props = self.frame.props

        self.frame_copy = self.frame.copy()
        self.props_rw = self.frame_copy.props

    def test_video_props(self):
        self.core.std.BlankClip().get_frame(0).props

    def test_audio_props(self):
        self.core.std.BlankAudio().get_frame(0).props

    def test_item_access(self):
        self.assertEqual(self.props["_DurationDen"], 24)
        with self.assertRaises(KeyError):
            self.props["_NonExistent"]

        with self.assertRaises(vs.Error):
            self.props["_DurationDen"] = 1

        with self.assertRaises(vs.Error):
            del self.props["_DurationDen"]

        self.assertEqual(self.props["_DurationDen"], 24)

        self.assertEqual(self.props_rw["_DurationDen"], 24)
        self.props_rw["_DurationDen"] = 1
        self.assertEqual(self.props_rw["_DurationDen"], 1)
        del self.props_rw["_DurationDen"]
        self.assertFalse("_DurationDen" in self.props_rw)

    def test_length(self):
        self.assertCountEqual(self.props, self.props_rw)
        self.assertEqual(len(self.props_rw), 2)
        del self.props_rw["_DurationDen"]
        self.assertEqual(len(self.props_rw), 1)
        self.props_rw["_DurationDen"] = 1
        self.assertEqual(len(self.props_rw), 2)
        self.props_rw["TestEntry"] = "123"
        self.assertEqual(len(self.props_rw), 3)

    def test_iterators(self):
        self.assertEqual(list(self.props.keys()), list(self.props))
        self.assertEqual(self.props.keys(), set(["_DurationDen", "_DurationNum"]))
        self.assertEqual(list(self.props.values()), [24, 1])
        self.assertEqual(self.props.items(), {("_DurationDen", 24), ("_DurationNum", 1)})
        self.assertEqual(dict(self.props), dict(self.props_rw))
        self.assertEqual(dict(self.props), {"_DurationDen": 24, "_DurationNum": 1})

    def test_get_pop(self):
        self.assertEqual(self.props.get("_DurationDen"), 24)
        self.assertEqual(self.props.get("_NonExistent"), None)
        self.assertEqual(self.props.get("_NonExistent", "Testificate"), "Testificate")

        with self.assertRaises(KeyError):
            self.props.pop("_NonExistent")

        x = []
        self.assertTrue(self.props.pop("_NonExistent", x) is x)
        self.assertEqual(self.props.pop("_NonExistent", None), None)
        self.assertEqual(self.props.pop("_NonExistent", "Testificate"), "Testificate")

        self.assertEqual(self.props_rw.pop("_DurationDen"), 24)
        with self.assertRaises(KeyError):
            self.props_rw.pop("_DurationDen")
        self.assertEqual(self.props_rw.pop("_DurationDen", "Test"), "Test")
        self.assertEqual(self.props_rw.popitem(), ("_DurationNum", 1))

    def test_setdefault(self):
        self.assertEqual(self.props_rw.setdefault("_DurationDen"), 24)
        self.assertFalse("_NonExistent1" in self.props_rw)
        self.assertEqual(self.props_rw.setdefault("_NonExistent1"), 0)
        self.assertEqual(self.props_rw["_NonExistent1"], 0)

        self.assertEqual(self.props_rw.setdefault("_NonExistent2", b"Testificate"), b"Testificate")
        self.assertEqual(self.props_rw["_NonExistent2"], b"Testificate")

    def test_attr_access(self):
        self.assertEqual(self.props._DurationDen, 24)
        with self.assertRaises(AttributeError):
            self.props._NonExistent

        with self.assertRaises(vs.Error):
            self.props._DurationDen = 1

        with self.assertRaises(vs.Error):
            del self.props._DurationDen

        self.assertEqual(self.props._DurationDen, 24)

        self.assertEqual(self.props_rw._DurationDen, 24)
        self.props_rw._DurationDen = 1
        self.assertEqual(self.props_rw._DurationDen, 1)
        del self.props_rw._DurationDen
        self.assertFalse(hasattr(self.props_rw, "_DurationDen"))

    def test_data_props(self):
        self.props_rw.DataPropStr = "hello"
        self.props_rw.DataPropBytes = b"hello"
        self.assertEqual(type(self.props_rw.DataPropStr), str)
        self.assertEqual(type(self.props_rw.DataPropBytes), bytes)

    def test_no_nodes_or_functions(self):
        clip = self.core.std.BlankClip()
        with self.assertRaises(vs.Error):
            self.props_rw["node"] = clip
        with self.assertRaises(vs.Error):
            self.props_rw["func"] = lambda: None
        with self.assertRaises(vs.Error):
            self.props_rw["mixed"] = [1, clip]
        self.assertFalse("node" in self.props_rw)
        self.assertFalse("func" in self.props_rw)
        self.assertFalse("mixed" in self.props_rw)
        with self.assertRaises(vs.Error):
            self.core.std.SetFrameProps(clip, ref=clip)
        with self.assertRaises(vs.Error):
            self.core.std.SetFrameProps(clip, fn=lambda: None)

    def test_setting_a_property_survives_the_frame_closing_mid_iteration(self):
        # the values are converted before the native map is fetched, so a generator that closes
        # the frame between two values cannot leave the setter writing into a freed map
        frame = self.core.std.BlankClip().get_frame(0).copy()

        def values():
            yield 1
            frame.close()
            yield 2

        with self.assertRaises(RuntimeError):
            frame.props["closes_midway"] = values()
        self.assertTrue(frame.closed)

    def test_setting_a_property_rejects_a_value_that_closes_the_frame(self):
        # the same through a conversion rather than the iterator
        frame = self.core.std.BlankClip().get_frame(0).copy()

        class ClosesOnConversion(int):
            def __int__(self):
                frame.close()
                return 5

        with self.assertRaises(RuntimeError):
            frame.props["closes_on_convert"] = [1, ClosesOnConversion(2)]


if __name__ == "__main__":
    unittest.main()
