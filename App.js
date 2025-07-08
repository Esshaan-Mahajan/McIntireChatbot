// App.js
import React, { useState } from 'react';
import { SafeAreaView, View, Text, TextInput, Button, ScrollView, StyleSheet, ActivityIndicator } from 'react-native';
import axios from 'axios';

const App = () => {
  // States to hold message input and chat history
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  // Replace with your deployed Flask API URL.
  const API_URL = 'https://mcintire-chatbot-ed1de2ff18cc.herokuapp.com/chat';
;

  // Function to send a message to the API
  const sendMessage = async () => {
    if (!message) return;
    const userMsg = { sender: 'You', text: message };
    setChatHistory(prev => [...prev, userMsg]);
    setLoading(true);

    // Prepare form data for the POST request.
    // In this example, we are sending only text.
    const formData = new FormData();
    formData.append('text', message);
    // For a text-based conversation, we'll force output to "text"
    formData.append('output_type', 'text');

    try {
      // Post the data to your Flask API endpoint.
      const response = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const botMsg = { sender: 'Bot', text: response.data.response };
      setChatHistory(prev => [...prev, botMsg]);
    } catch (error) {
      const errorMsg = { sender: 'Bot', text: 'Error: ' + error.message };
      setChatHistory(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
      setMessage('');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.chatBox}>
        {chatHistory.map((msg, index) => (
          <View key={index} style={msg.sender === 'You' ? styles.userMessage : styles.botMessage}>
            <Text style={styles.messageText}>
              <Text style={{ fontWeight: 'bold' }}>{msg.sender}: </Text>
              {msg.text}
            </Text>
          </View>
        ))}
        {loading && <ActivityIndicator size="small" color="#0d6efd" />}
      </ScrollView>
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={message}
          onChangeText={setMessage}
          placeholder="Type your message..."
        />
        <Button title="Send" onPress={sendMessage} />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  chatBox: { flex: 1, padding: 10 },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#0d6efd',
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#e9ecef',
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
  },
  messageText: { color: '#000' },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    borderTopWidth: 1,
    borderColor: '#ccc',
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    marginRight: 10,
    paddingHorizontal: 10,
  },
});

export default App;
