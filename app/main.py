try:
    from app.ui.streamlit_app import run_app
except ModuleNotFoundError:
    from ui.streamlit_app import run_app


if __name__ == "__main__":
    run_app()
