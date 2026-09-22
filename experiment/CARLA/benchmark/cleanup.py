"""Attempt every teardown step and retain failures without masking run errors."""
import json


class CleanupJournal:
    def __init__(self):
        self.steps=[]

    def attempt(self, name, action):
        try:
            result=action()
            if result is False:
                raise RuntimeError('cleanup API returned False')
            self.steps.append(dict(step=name,status='COMPLETED'))
        except Exception as error:
            self.steps.append(dict(step=name,status='FAILED',error_type=type(error).__name__,error=str(error)))

    def result(self):
        return dict(completed=all(step['status']=='COMPLETED' for step in self.steps),steps=list(self.steps),
                    scope='teardown API calls; not proof that the server has no unrelated actors')

    def save(self, path):
        try:
            path.write_text(json.dumps(self.result(),indent=2),encoding='utf-8')
        except Exception as error:
            self.steps.append(dict(step='write_cleanup_report',status='FAILED',error_type=type(error).__name__,error=str(error)))
        return self.result()
