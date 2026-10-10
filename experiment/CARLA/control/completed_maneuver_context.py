"""Resolve post-maneuver language only from observed completed execution."""
from copy import deepcopy
import re

ZH=re.compile(r'^\s*(?:完成|结束)(?P<side>左|右)转(?:之)?后[，,：:\s]*(?P<tail>.+)$')
EN=re.compile(r'^\s*(?:after|once)\s+(?:(?:completing|finishing)\s+)?(?:the\s+)?'
    r'(?P<side>left|right)(?:-hand)?\s+turn\s*[,;:]\s*(?P<tail>.+)$',re.I)


class CompletedManeuverContext:
    def __init__(self):
        self.key=None
        self.resolved=None

    def resolve(self, command, previous_plan):
        text=str(command.get('text') or command.get('source_text') or command.get('voice_text')
            or command.get('normalized_text') or command.get('parser_text_en') or '')
        key=(command.get('id'),text,command.get('parser_text_en'))
        if key==self.key and self.resolved is not None:
            result=deepcopy(dict(command))
            result.update(deepcopy(self.resolved))
            return result
        self.key=key;self.resolved=None
        match=ZH.match(text) or EN.match(text)
        state=getattr(previous_plan,'state',None) or {}
        document=getattr(previous_plan,'document',None) or {}
        if not match or state.get('plan_status')!='COMPLETED' or document.get('request_id')==command.get('id'):
            return command
        side={'左':'LEFT','右':'RIGHT','left':'LEFT','right':'RIGHT'}[match['side'].lower()]
        states=state.get('step_states') or []
        steps=(document.get('intent') or {}).get('steps') or []
        if not states or not steps:
            return command
        observed=states[-1];step=steps[-1]
        if (observed.get('status')!='COMPLETED' or observed.get('action')!='TURN'
                or observed.get('step_id')!=step.get('step_id') or step.get('action')!='TURN'
                or (step.get('parameters') or {}).get('direction')!=side
                or not observed.get('terminal_frame_id')
                or 'observed_physical_completion' not in (observed.get('reason_codes') or [])):
            return command
        english=command.get('parser_text_en')
        english_match=EN.match(english) if isinstance(english,str) else None
        if english_match and english_match['side'].upper()!=side:
            return command
        tail=re.sub(r'^\s*(?:再|然后|接着)\s*','',match['tail']).strip()
        if not tail:
            return command
        result=deepcopy(dict(command))
        result.update(text=tail,original_source_text=text,
            command_history_resolution=dict(source='observed_completed_plan',
                previous_request_id=document.get('request_id'),previous_step_id=observed['step_id'],
                completion_frame_id=observed['terminal_frame_id'],completed_direction=side,
                original_text=text,active_text=tail))
        if english_match:
            result['parser_text_en']=english_match['tail'].strip()
        self.resolved={key:result[key] for key in ('text','original_source_text','command_history_resolution')}
        if english_match:self.resolved['parser_text_en']=result['parser_text_en']
        return deepcopy(result)
