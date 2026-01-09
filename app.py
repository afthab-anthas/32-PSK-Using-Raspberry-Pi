## all comments are written by our group members for easy understanding
from flask import Flask, request, render_template
import json
import re ##regulate library for cleaning the user input
import socket

app = Flask(__name__)

MATLAB_HOST = "afthab.local"
MATLAB_PORT = 5174


def parse_bitstream(raw_text):

    bits_str_list = re.findall(r'[01]', raw_text)
    return [int(b) for b in bits_str_list] ##return all chars that match with 0 and 1 from the user input


def send_and_receive_matlab(bit_list): ##bit_lists is the array of bits after cleaning
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) ## creating a tcp socket object which we will use to connect to matlab
        s.settimeout(60)
        s.connect((MATLAB_HOST, MATLAB_PORT)) ##initiates the TCP connection with the matlab ip and the port number


        payload = json.dumps({"flat_bits": bit_list}) ##convert my stringg of integers into a json dictionary
        s.sendall(payload.encode()) ##sends that data by encoding (converting into bytes)


        received_data = b"" ##just a new intialised variable to store the json data
        while True: ##loop to run through till all the plots and data are receieved as we get chunks of bytes, not everything at once
            try:
                chunk = s.recv(4096) ## takes 4096 bytes from matlab server
                if not chunk:
                    break
                received_data += chunk ## appending all the bits together
                s.settimeout(1.0)
            except socket.timeout:
                break

        s.close()

        if not received_data:
            return None

        # 3. Decode JSON
        return json.loads(received_data.decode('utf-8')) ## first the received bytes are converted to strings of utf-8 (char encode) which is then converted to a py dictionary

    except Exception as e: ##exception variable has the error value received from the json req
        print(f"Error communicating with MATLAB: {e}")
        return None ##error handling


@app.route("/", methods=["GET"]) ## routing to / (index file) which says is / is there in url, redirect it to index page (GET (RECEIVE DATA) REQUEST)
def index():
    return render_template("index.html") ##displays index.html


@app.route("/submit", methods=["POST"]) ##same thing as above but after you click submit form button (POST (SEND DATA) REQUEST)
def submit_bits():
    raw = request.form.get("bitstream", "").strip() ## finds the value inside the variable called bitstream from the form from " to the next "
    flat_bits = parse_bitstream(raw) ## uses the user defined function the clean the input to 0's and 1's

    if len(flat_bits) == 0:
        return render_template("index.html", error="No valid bits found.")

    if len(flat_bits) % 5 != 0:
        return render_template("index.html", error=f"Invalid length. Please enter inputs in multiples of 5")


    results = send_and_receive_matlab(flat_bits)

    if results and results.get("status") == "success":
        return render_template("results.html", data=results) ##store the received json data into the variable data (data.ser, data.ber, data.plots) and send it as input to results.html
    else:
        return render_template("index.html", error="Failed to get response from MATLAB.")
 

if __name__ == "__main__": ##user specifying that this is the main file and this should be run directly
    app.run(host="0.0.0.0", port=5173, debug=True) ##telling rpi to run the app on 0.0.0.0 for all users to see it and port should be 5173
