"""Controlled replay of numerical choices; reuse exact model tests without loosening gates."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from scripts.full_project import study
from scripts.compile_boundary.common import baseline, ROOT

EVIDENCE=study.OUT/'variants'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(tag):
    assert tag in ('v1_fp32_intermediate','v2_bf16_boundary')
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    dest=EVIDENCE/f'numerical_{tag}.json'
    assert not dest.exists(), 'Never overwrite a measured variant'
    assert not (study.OUT/'numerical_fused.json').exists(), 'Final measurement already exists'
    import src.qk_norm_rope as qk
    source=ROOT/'src/qk_norm_rope/kernel.cu'
    original_fused=qk.fused
    if tag=='v1_fp32_intermediate':
        variant_source=EVIDENCE/'variants'/tag/'kernel.cu'
        variant_source.parent.mkdir(parents=True,exist_ok=True)
        text=source.read_text()
        old='return __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(x,inv),w)));'
        assert text.count(old)==1
        variant_source.write_text(text.replace(old,'return __fmul_rn(__fmul_rn(x,inv),w);'))
        from torch.utils.cpp_extension import load
        build_dir=Path.home()/'.cache/nano-vllm-fusion-learning/replay_20260921_v1'
        build_dir.mkdir(parents=True,exist_ok=True)
        extension=load(name='qk_replay_20260921_v1',sources=[str(variant_source)],build_directory=str(build_dir),
            extra_cuda_cflags=['-O3','-lineinfo','--fmad=false'],extra_cflags=['-O3'],verbose=False)
        qk.load_extension=lambda:extension
        qk.build=lambda:extension
        source=variant_source
    # Both diagnostic versions use the same standalone B1 reduction organization.
    # The only v1→v2 change is restoration of the BF16 intermediate boundary.
    def candidate(*args,**kwargs):
        kwargs['reduction_mode']='standalone'
        return original_fused(*args,**kwargs)
    qk.fused=candidate
    from scripts.full_project.adapter import install
    install()
    failure=None
    try:
        study.numerical('fused')
    except AssertionError as exc:
        if str(exc)!='Full-model exact gate failed; diagnostic JSON retained':raise
        failure=str(exc)
    output=study.OUT/'numerical_fused.json'
    result=json.loads(output.read_text())
    assert result['complete'] and len(result['rows'])==7
    result.update(experiment_variant=tag,kernel_source=str(source),kernel_sha256=sha(source),
        driver_sha256=sha(Path(__file__)),reduction_mode='standalone',
        intermediate_bf16_rounding=tag=='v2_bf16_boundary',expected_result_predeclared=False,
        gate_failure=failure,performance_claim=False)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    # Preserve the unmodified raw output and free the canonical final-output name.
    output.rename(EVIDENCE/f'raw_numerical_{tag}.json')
    print('VARIANT RESULT',tag,'exact_gate=',result['exact_gate'],
          'exact_steps=',sum(r['exact_steps'] for r in result['rows']),flush=True)

if __name__=='__main__':main(sys.argv[1])
