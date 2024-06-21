import summ_eval
import pkgutil

def safe_import(name):
    try:
        return __import__(f"summ_eval.{name}", fromlist=[name])
    except Exception:
        return None

def discover_metrics():
    module_names = [name for _, name, _ in pkgutil.iter_modules(summ_eval.__path__)]
    modules = [safe_import(name) for name in module_names]
    return {
        module_name: dir(module)
        for module_name, module in zip(module_names, modules)
    }

if __name__ == "__main__":
    metrics = discover_metrics()
    print(metrics)