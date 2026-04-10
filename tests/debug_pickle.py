import jax
import jax.numpy as jnp
import cloudpickle
import jax._src.core as core
import traceback

def strip_all_tracebacks(obj):
    if isinstance(obj, dict):
        return {k: strip_all_tracebacks(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_all_tracebacks(x) for x in obj]
    if isinstance(obj, tuple):
        if hasattr(obj, '_replace') and hasattr(obj, 'traceback'):
            obj = obj._replace(traceback=None)
        return tuple(strip_all_tracebacks(x) for x in obj)
    if isinstance(obj, core.Jaxpr):
        return strip_source_info(obj)
    if hasattr(obj, 'jaxpr') and hasattr(obj, 'consts'): # ClosedJaxpr
        return core.ClosedJaxpr(strip_source_info(obj.jaxpr), strip_all_tracebacks(obj.consts))
    return obj

def strip_source_info(jaxpr):
    new_eqns = []
    for eqn in jaxpr.eqns:
        new_params = strip_all_tracebacks(eqn.params)
        si = None # Force None
        
        if hasattr(eqn, '_replace'):
            new_eqn = eqn._replace(params=new_params, source_info=si)
        else:
            new_eqn = eqn # Should not happen for JaxprEqn
        new_eqns.append(new_eqn)
        
    return core.Jaxpr(
        constvars=jaxpr.constvars,
        invars=jaxpr.invars,
        outvars=jaxpr.outvars,
        eqns=new_eqns
    )

def f(x):
    return jnp.sin(jnp.cos(x)) * x

def run_debug():
    print("Creating Jaxpr...")
    closed = jax.make_jaxpr(f)(jnp.ones(10))
    jaxpr = closed.jaxpr
    
    print("Stripping tracebacks...")
    clean = strip_all_tracebacks(jaxpr)
    
    print("Drilling down...")
    if not clean.eqns:
        print("No equations in Jaxpr!")
        return

    eqn = clean.eqns[0]
    print(f"Eqn type: {type(eqn)}")
    if hasattr(eqn, '_fields'):
        print(f"Eqn fields: {eqn._fields}")
    
    print(f"Eqn dir: {dir(eqn)}")
    
    ctx = getattr(eqn, 'ctx', None)
    if ctx:
        print(f"Eqn ctx: {ctx}")
        print(f"Eqn ctx type: {type(ctx)}")
        try:
            cloudpickle.dumps(ctx)
            print("ctx OK")
        except: print("ctx FAIL")
        
        # Try constructor pos 1
        try:
            core.JaxprEqn(
                eqn.invars, eqn.outvars, eqn.primitive, eqn.params, eqn.effects, None, ctx
            )
            print("Constructor (..., source_info, ctx) SUCCESS")
        except Exception as e:
            print(f"Constructor (..., source_info, ctx) FAILED: {e}")
            
        # Try constructor pos 2 (reverse?)
        try:
            core.JaxprEqn(
                ctx, eqn.invars, eqn.outvars, eqn.primitive, eqn.params, eqn.effects, None
            )
            print("Constructor (ctx, ...) SUCCESS")
        except Exception as e:
            print(f"Constructor (ctx, ...) FAILED: {e}")
            
        # Try constructor pos 3 (after params?)
        try:
            core.JaxprEqn(
                 eqn.invars, eqn.outvars, eqn.primitive, eqn.params, ctx, eqn.effects, None
            )
            print("Constructor (..., params, ctx, effects, ...) SUCCESS")
        except Exception as e:
             print(f"Constructor (..., params, ctx, effects, ...) FAILED: {e}")

    print(f"Eqn source_info: {eqn.source_info}")

    try:
        cloudpickle.dumps(eqn.params)
        print("Params OK")
    except: print("Params FAIL")

    try:
        cloudpickle.dumps(eqn.invars)
        print("Invars OK")
    except: print("Invars FAIL")

    try:
        cloudpickle.dumps(eqn.outvars)
        print("Outvars OK")
    except: print("Outvars FAIL")

    try:
        cloudpickle.dumps(eqn.source_info)
        print(f"SourceInfo: {eqn.source_info}")
        if eqn.source_info is None: print("SourceInfo is None (OK)")
        else: print("SourceInfo type:", type(eqn.source_info))
    except: print("SourceInfo FAIL")
    
    # Check vars for debug info
    try:
        v = eqn.invars[0]
        print("Var dir:", dir(v))
    except: pass

    try:
        cloudpickle.dumps(clean)
        print("Success!")
    except Exception as e:
        print(f"Pickle failed: {e}")

if __name__ == "__main__":
    run_debug()
