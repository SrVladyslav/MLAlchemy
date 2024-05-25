#!/bin/bash

echo "Activating virtual environment"

activate () {
    source "./../.env/Scripts/activate"
    echo "Activated!"
}
activate