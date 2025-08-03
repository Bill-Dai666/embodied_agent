from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from .views import skill
import os

def create_app():
  app = FastAPI(
    title="Skill Hooks",
    version="0.0.0",
    description="Your description goes here",
  )

  # Add API route
  app.include_router(router=skill.router)

  # Setup template rendering
  # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  # templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

  # # Add HTML route
  # @app.get("/", response_class=HTMLResponse)
  # async def serve_home(request: Request):
  #     return templates.TemplateResponse("index.html", {"request": request})

  return app