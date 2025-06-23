from flask import Flask, Response, render_template
import test

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run-script')
def run_script():
    def generate():
        for output in test.data_source():
            yield f"data: {output}\n\n"  # Format for Server-Sent Events (SSE)

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
