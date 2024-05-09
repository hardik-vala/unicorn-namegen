import streamlit as st
from sample import sample_names


def main():
    st.title("🦄 Unicorn Name Generator")

    num_samples = int(st.select_slider(
      "No. names to sample",
      options=map(str, list(range(1, 26))),
      value="10"))

    if st.button("Generate"):
        st.divider()
        with st.spinner("Generating names..."):
            names = sample_names(num_samples)
        for i, n in enumerate(names):
            st.markdown(f"{i + 1}. **{n}**")


if __name__ == "__main__":
    main()
