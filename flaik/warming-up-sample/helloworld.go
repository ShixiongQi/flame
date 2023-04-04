package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
	"io/ioutil"
)

func handler(w http.ResponseWriter, r *http.Request) {
	log.Print("helloworld: received a request")
	target := os.Getenv("TARGET")
	if target == "" {
		target = "World"
	}
	fmt.Fprintf(w, "Hello %s!\n", target)
}

func sync_sleep_handler(w http.ResponseWriter, r *http.Request) {
	log.Print("Synchronous sleep received a request.")

	pauseTime := 180 * time.Second
	time.Sleep(pauseTime)

	target := os.Getenv("TARGET")
	if target == "" {
		target = "World"
	}
	fmt.Fprintf(w, "Synchronous sleep %s!\n", target)
}

func async_sleep_handler(w http.ResponseWriter, r *http.Request) {
	log.Print("Asynchronous sleep received a request.")

	go func() {
		pauseTime := 360 * time.Second
		time.Sleep(pauseTime)
	}()

	target := os.Getenv("TARGET")
	if target == "" {
		target = "World"
	}
	fmt.Fprintf(w, "Asynchronous sleep %s!\n", target)
}

func async_warmup_handler(w http.ResponseWriter, r *http.Request) {
	log.Print("Asynchronous warm-up received a request.")

	go func() {
		client := &http.Client{
			Timeout: time.Second * 60,
		}
		req, err := http.NewRequest("GET", "http://10.10.1.1:30918/", nil)
		if err != nil {
			fmt.Errorf("Got error %s", err.Error())
		}

		req.Host = "warm-up-test.default.example.com"

		for {
			log.Print("Sending a warm-up request.")
			response, err := client.Do(req)
			if err != nil {
				fmt.Errorf("Got error %s", err.Error())
			}
			defer response.Body.Close()
			body, err := ioutil.ReadAll(response.Body)
			if err != nil {
				panic(err)
			}

			log.Printf("Response StatusCode: %v", response.StatusCode)
			log.Printf("Response Body: %v", string(body))

			pauseTime := 5 * time.Second
			time.Sleep(pauseTime)
		}
		
	}()

	target := os.Getenv("TARGET")
	if target == "" {
		target = "World"
	}
	fmt.Fprintf(w, "Asynchronous warm-up %s!\n", target)
}

func main() {
	log.Print("helloworld: starting server...")

	http.HandleFunc("/", handler)
	http.HandleFunc("/1", sync_sleep_handler)
	http.HandleFunc("/2", async_sleep_handler)
	http.HandleFunc("/3", async_warmup_handler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("helloworld: listening on port %s", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}
