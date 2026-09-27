import unittest

from control.command_dispatch import CompletionCommandQueue


class DispatchWaitBudgetTests(unittest.TestCase):
    def queue(self, **kwargs):
        return CompletionCommandQueue([{'driving_intent': {
            'request_id': 'request', 'intent': {'steps': [{'action': 'SET_SPEED'}]}}}], **kwargs)

    def feedback(self, timestamp, **kwargs):
        return dict(request_id='request', plan_status='ACTIVE',
                    observed_at_s=timestamp, safety_wait=True, **kwargs)

    def test_fresh_safety_wait_preserves_execution_budget(self):
        q=self.queue(timeout_s=1)
        q.select(0,0)
        for i in range(1,21):
            q.select(0,i*.1,self.feedback((i-1)*.1))
        self.assertAlmostEqual(q.execution_elapsed_s,0)
        self.assertAlmostEqual(q.safety_wait_elapsed_s,2)
        q.select(1,2.1)
        self.assertAlmostEqual(q.execution_elapsed_s,.1)

    def test_wait_is_bounded(self):
        q=self.queue(timeout_s=10,safety_wait_timeout_s=1)
        q.select(0,0)
        for i in range(1,4):
            q.select(0,i*.25,self.feedback((i-1)*.25))
        with self.assertRaisesRegex(ValueError,'SAFETY_WAIT_TIMEOUT'):
            q.select(0,1,self.feedback(.75))

    def test_stale_or_wrong_request_cannot_pause(self):
        for changed in ({'observed_at_s': -1}, {'request_id':'other'}, {'safety_wait':False}):
            q=self.queue(timeout_s=.5)
            q.select(0,0)
            feedback=self.feedback(0)
            feedback.update(changed)
            with self.assertRaisesRegex(ValueError,'TIMEOUT'):
                q.select(0,.5,feedback)

    def test_completion_before_execution_deadline(self):
        q=self.queue(timeout_s=1)
        q.select(0,0)
        q.select(1,.9,{'request_id':'request','plan_status':'COMPLETED'})
        self.assertEqual(q.status()['completed_commands'],1)

    def test_window_expiry_still_fails(self):
        q=self.queue(timeout_s=10)
        q.commands[0]['_dispatch_end_m']=1
        q.select(0,0)
        with self.assertRaisesRegex(ValueError,'ACTIVE_COMMAND_WINDOW_EXPIRED'):
            q.select(1,.1,self.feedback(0))


if __name__=='__main__':
    unittest.main()
