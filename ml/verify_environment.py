"""Import each runtime dependency and print its installed version."""

from importlib import import_module

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "streamlit": "streamlit",
}


def main() -> int:
    missing = []
    for package, module_name in MODULES.items():
        try:
            module = import_module(module_name)
        except ImportError as exc:
            print(f"MISSING {package}: {exc}")
            missing.append(package)
        else:
            print(f"{package}: {getattr(module, '__version__', 'version unavailable')}")

    if missing:
        print("Environment verification failed. Missing: " + ", ".join(missing))
        return 1
    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
