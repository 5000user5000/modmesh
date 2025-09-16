# Copyright (c) 2019, Yung-Yu Chen <yyc@solvcon.net>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import unittest
import time

import modmesh


class StopWatchTC(unittest.TestCase):

    def test_singleton(self):

        self.assertIs(modmesh.stop_watch, modmesh.StopWatch.me)

    def test_microsecond_resolution(self):

        sw = modmesh.stop_watch
        self.assertGreater(1.e-6, sw.resolution)

    @unittest.skipUnless("nt" != os.name,
                         "timing code on windows does not work yet")
    def test_lap_with_sleep(self):

        sw = modmesh.stop_watch

        # Mark start
        sw.lap()

        time.sleep(0.01)

        elapsed = sw.lap()
        self.assertGreater(elapsed, 0.01)
        # Don't test for the upper bound. CI doesn't like it (to be specific,
        # mac runner of github action).


class WrapperProfilerStatusTC(unittest.TestCase):

    def test_singleton(self):

        self.assertIs(
            modmesh.wrapper_profiler_status,
            modmesh.WrapperProfilerStatus.me)

    def test_default(self):

        self.assertTrue(modmesh.wrapper_profiler_status.enabled)


class TimeRegistryTC(unittest.TestCase):

    _profiler_enabled_default = modmesh.wrapper_profiler_status.enabled

    def setUp(self):

        modmesh.wrapper_profiler_status.enable()

    def tearDown(self):

        if self._profiler_enabled_default:
            modmesh.wrapper_profiler_status.enable()
        else:
            modmesh.wrapper_profiler_status.disable()

    def test_singleton(self):

        self.assertIs(modmesh.time_registry, modmesh.TimeRegistry.me)

    def test_empty_report(self):

        modmesh.time_registry.clear()
        ret = modmesh.time_registry.report()
        self.assertEqual("", ret)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
