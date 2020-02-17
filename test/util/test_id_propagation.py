import unittest


class TestIdPropagation(unittest.TestCase):
    root = '../../data'

    # FIXME this does not work right now
    @unittest.expectedFailure
    def test_ids_fig2b(self):
        from mmpb.util import propagate_ids
        # Ids for Fig 2B, from v 0.5.5 to v 0.6.6 (=1.0.0)
        old_ids = [4136, 4645, 4628, 3981, 2958, 3108, 4298]
        expected_ids = [4211, 4724, 4707, 3031, 4056, 3181, 4373]

        old_version = '0.5.5'
        new_version = '0.6.6'
        name = 'sbem-6dpf-1-whole-segmented-cells'
        mapped_ids = propagate_ids(self.root, old_version, new_version,
                                   name, old_ids)
        self.assertEqual(len(expected_ids), len(mapped_ids))
        self.assertEqual(set(expected_ids), set(mapped_ids))

    # FIXME this does not work right now
    @unittest.expectedFailure
    def test_ids_fig2c(self):
        from mmpb.util import propagate_ids
        # Ids for Fig 2C from v 0.3.1  to v 0.6.6 (=1.0.0)
        old_ids = [1350, 5312, 5525, 5720, 6474, 6962, 7386,
                   8143, 8144, 8177, 8178, 8885, 10027, 11092]
        expected_ids = [1425, 5385, 5598, 5795, 6552, 7044, 7468, 8264,
                        8230, 8231, 8987, 9185, 10167, 11273]

        old_version = '0.3.1'
        new_version = '0.6.6'
        name = 'sbem-6dpf-1-whole-segmented-cells'
        mapped_ids = propagate_ids(self.root, old_version, new_version,
                                   name, old_ids)
        self.assertEqual(len(expected_ids), len(mapped_ids))
        self.assertEqual(set(expected_ids), set(mapped_ids))

    # test propagation between a single version: 0.6.4 -> 0.6.5
    def test_v64_to_v65(self):
        from mmpb.util import propagate_ids
        # TODO get more id pairs from platy browser
        old_ids = [5787,  # neuropil
                   ]
        expected_ids = [5773,  # neuropil
                        ]

        old_version = '0.6.4'
        new_version = '0.6.5'
        name = 'sbem-6dpf-1-whole-segmented-cells'
        mapped_ids = propagate_ids(self.root, old_version, new_version,
                                   name, old_ids)
        print(expected_ids, mapped_ids)
        self.assertEqual(len(expected_ids), len(mapped_ids))
        self.assertEqual(set(expected_ids), set(mapped_ids))


if __name__ == '__main__':
    unittest.main()
