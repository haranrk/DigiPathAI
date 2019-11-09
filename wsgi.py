from DigiPathAI.main_server import app

if __name__ == "__main__":
    app.config['SLIDE_DIR'] = 'examples'
    app.viewer_only = True
    app.run()
