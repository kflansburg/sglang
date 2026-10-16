# Verbatim-exec of HostKVCache (pool_host/base.py HEAD) alloc/free/clear.
# Run from repo root: python3 audit/repros/repro_f2_sentinel_free.py
import re, torch, threading, typing
def synchronized(func):
    def wrapper(self,*a,**k):
        with self.lock: return func(self,*a,**k)
    return wrapper
src=open('python/sglang/srt/mem_cache/pool_host/base.py').read()
alloc=re.search(r'    @synchronized\n    def alloc\(.*?(?=\n    @synchronized\n    def free)',src,re.S).group(0)
free=re.search(r'    @synchronized\n    def free\(.*?\Z',src,re.S).group(0)
clear=re.search(r'    @synchronized\n    def clear\(.*?(?=\n    def available_size)',src,re.S).group(0)
avail=re.search(r'    def available_size\(.*?(?=\n    def _merge)',src,re.S).group(0)
merge=re.search(r'    def _merge_release_slots\(.*?(?=\n    @property)',src,re.S).group(0)
props=('    @property\n    def logical_size(self):\n        return self.size\n'
       '    @property\n    def logical_page_size(self):\n        return self.page_size\n')
code="class FakePool:\n"+props+clear+avail+merge+alloc+free
ns={'torch':torch,'threading':threading,'synchronized':synchronized,'Optional':typing.Optional}
exec(code,ns)
f=ns['FakePool'](); f.size=128; f.page_size=64; f.lock=threading.RLock(); f.device='cpu'; f.clear()
live=f.alloc(128)
bad=torch.full((64,),-1,dtype=torch.int64); bad[0]=5
f.free(bad)  # no assert raised
assert not f.slot_used[127], 'live slot 127 silently marked free'
f._merge_release_slots()
nxt=f.alloc(64)
assert (nxt<0).sum().item()==63 and (nxt==5).sum().item()==1
print('F2 CONFIRMED: -1 sentinels issued as live slot ids; live slot 127 ownership stolen')
