import yappi
import sys


def getDataAbout(module_list_names, threads=1):

    modules = [sys.modules.get(module_name) for module_name in module_list_names]

    if len(modules):
        stats = yappi.get_func_stats(
            filter_callback=lambda x: yappi.module_matches(x, modules)
        )
    else:
        stats = yappi.get_func_stats()

    return stats
