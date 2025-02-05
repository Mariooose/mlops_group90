import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/pivotal-base-447808-q9/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    print(services)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"

    response = requests.post(predict_url, files={"data": image}, timeout=10)

    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Pokémon Classification")
    # st.write("Upload a PNG of a pokemon and our amazing ML model will tell which pokemon it is!")
    st.write("Who's that Pokémon?")

    uploaded_file = st.file_uploader("Upload an image", type=["png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            predictions = result["pred1"]
            probabilities = result["prob1"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Its ", predictions[0], "!")

            st.write("Top 5 predictions:")
            st.write("1: Prediction: ", predictions[0], "Confidence: ", probabilities[0])
            st.write("2: Prediction: ", predictions[1], "Confidence: ", probabilities[1])
            st.write("3: Prediction: ", predictions[2], "Confidence: ", probabilities[2])
            st.write("4: Prediction: ", predictions[3], "Confidence: ", probabilities[3])
            st.write("5: Prediction: ", predictions[4], "Confidence: ", probabilities[4])

            # make a nice bar chart
            # data = {"Class": [f"Class {i}" for i in range(10)], "Probability": probabilities}
            # df = pd.DataFrame(data)
            # df.set_index("Class", inplace=True)
            # st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
