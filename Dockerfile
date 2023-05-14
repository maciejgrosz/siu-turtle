FROM dudekw/siu-20.04

WORKDIR /root

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python ./get-pip.py 
RUN rm -f ./get-pip.py

#Install TensorFlow	
#RUN pip install --no-cache-dir TensorFlow	
#RUN apt-get install TensorFlow
COPY roads.png /roads.png
# RUN source /root/siu_ws/devel/setup.bash

# ENTRYPOINT ["tail", "-f", "/dev/null"]